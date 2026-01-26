from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from ltx_mlx.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType, TransformerConfig
from ltx_mlx.models.ltx.adaln import AdaLayerNormSingle
from ltx_mlx.models.ltx.rope import precompute_freqs_cis
from ltx_mlx.models.ltx.text_projection import PixArtAlphaTextProjection
from ltx_mlx.models.ltx.transformer import BasicAVTransformerBlock, Modality, TransformerArgs
from ltx_mlx.utils import to_denoised


class TransformerArgsPreprocessor:
    def __init__(self, patchify_proj: nn.Linear, adaln: AdaLayerNormSingle, caption_projection: PixArtAlphaTextProjection, inner_dim: int, max_pos: List[int], num_attention_heads: int, use_middle_indices_grid: bool, timestep_scale_multiplier: int, positional_embedding_theta: float, rope_type: LTXRopeType, double_precision_rope: bool = False):
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.caption_projection = caption_projection
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope

    def _prepare_timestep(self, timestep: mx.array, batch_size: int, hidden_dtype: mx.Dtype = None) -> Tuple[mx.array, mx.array]:
        timestep = timestep * self.timestep_scale_multiplier
        timestep_emb, embedded_timestep = self.adaln(timestep.reshape(-1), hidden_dtype=hidden_dtype)
        timestep_emb = mx.reshape(timestep_emb, (batch_size, -1, timestep_emb.shape[-1]))
        embedded_timestep = mx.reshape(embedded_timestep, (batch_size, -1, embedded_timestep.shape[-1]))
        return timestep_emb, embedded_timestep

    def _prepare_context(self, context: mx.array, x: mx.array, attention_mask: Optional[mx.array] = None) -> Tuple[mx.array, Optional[mx.array]]:
        batch_size = x.shape[0]
        context = self.caption_projection(context)
        context = mx.reshape(context, (batch_size, -1, x.shape[-1]))
        return context, attention_mask

    def _prepare_attention_mask(self, attention_mask: Optional[mx.array], x_dtype: mx.Dtype) -> Optional[mx.array]:
        if attention_mask is None:
            return None
        if attention_mask.dtype in [mx.float16, mx.float32, mx.bfloat16]:
            return attention_mask
        mask = (attention_mask.astype(x_dtype) - 1) * 1e9
        mask = mx.reshape(mask, (attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
        return mask

    def _prepare_positional_embeddings(self, positions: mx.array, inner_dim: int, max_pos: List[int], use_middle_indices_grid: bool, num_attention_heads: int) -> Tuple[mx.array, mx.array]:
        pe = precompute_freqs_cis(positions, dim=inner_dim, theta=self.positional_embedding_theta, max_pos=max_pos, use_middle_indices_grid=use_middle_indices_grid, num_attention_heads=num_attention_heads, rope_type=self.rope_type, double_precision=self.double_precision_rope)
        return pe

    def prepare(self, modality: Modality) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        timestep, embedded_timestep = self._prepare_timestep(modality.timesteps, x.shape[0], hidden_dtype=x.dtype)
        context, attention_mask = self._prepare_context(modality.context, x, modality.context_mask)
        attention_mask = self._prepare_attention_mask(attention_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(positions=modality.positions, inner_dim=self.inner_dim, max_pos=self.max_pos, use_middle_indices_grid=self.use_middle_indices_grid, num_attention_heads=self.num_attention_heads)
        return TransformerArgs(x=x, context=context, context_mask=attention_mask, timesteps=timestep, embedded_timestep=embedded_timestep, positional_embeddings=pe, cross_positional_embeddings=None, cross_scale_shift_timestep=None, cross_gate_timestep=None, enabled=modality.enabled)


class LTXModel(nn.Module):
    def __init__(self, config: LTXModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.use_middle_indices_grid = config.use_middle_indices_grid
        self.rope_type = config.rope_type
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.positional_embedding_theta = config.positional_embedding_theta
        cross_pe_max_pos = None
        if config.model_type.is_video_enabled():
            self.positional_embedding_max_pos = config.positional_embedding_max_pos
            self.num_attention_heads = config.num_attention_heads
            self.inner_dim = config.inner_dim
            self._init_video(config)
        if config.model_type.is_audio_enabled():
            self.audio_positional_embedding_max_pos = config.audio_positional_embedding_max_pos
            self.audio_num_attention_heads = config.audio_num_attention_heads
            self.audio_inner_dim = config.audio_inner_dim
            self._init_audio(config)
        if config.model_type.is_video_enabled() and config.model_type.is_audio_enabled():
            cross_pe_max_pos = max(config.positional_embedding_max_pos[0], config.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = config.av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = config.audio_cross_attention_dim
            self._init_audio_video(config)
        self._init_preprocessors(config, cross_pe_max_pos)
        self._init_transformer_blocks(config)

    def _init_video(self, config: LTXModelConfig) -> None:
        self.patchify_proj = nn.Linear(config.in_channels, self.inner_dim, bias=True)
        self.adaln_single = AdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(in_features=config.caption_channels, hidden_size=self.inner_dim)
        self.scale_shift_table = mx.zeros((2, self.inner_dim))
        self.norm_out = nn.LayerNorm(self.inner_dim, eps=config.norm_eps, affine=False)
        self.proj_out = nn.Linear(self.inner_dim, config.out_channels)

    def _init_audio(self, config: LTXModelConfig) -> None:
        self.audio_patchify_proj = nn.Linear(config.audio_in_channels, self.audio_inner_dim, bias=True)
        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim)
        self.audio_caption_projection = PixArtAlphaTextProjection(in_features=config.audio_caption_channels, hidden_size=self.audio_inner_dim)
        self.audio_scale_shift_table = mx.zeros((2, self.audio_inner_dim))
        self.audio_norm_out = nn.LayerNorm(self.audio_inner_dim, eps=config.norm_eps, affine=False)
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, config.audio_out_channels)

    def _init_audio_video(self, config: LTXModelConfig) -> None:
        num_scale_shift_values = 4
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=num_scale_shift_values)
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=num_scale_shift_values)
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=1)
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=1)

    def _init_preprocessors(self, config: LTXModelConfig, cross_pe_max_pos: Optional[int]) -> None:
        if config.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(patchify_proj=self.patchify_proj, adaln=self.adaln_single, caption_projection=self.caption_projection, inner_dim=self.inner_dim, max_pos=config.positional_embedding_max_pos, num_attention_heads=self.num_attention_heads, use_middle_indices_grid=config.use_middle_indices_grid, timestep_scale_multiplier=config.timestep_scale_multiplier, positional_embedding_theta=config.positional_embedding_theta, rope_type=config.rope_type, double_precision_rope=config.double_precision_rope)
        if config.model_type.is_audio_enabled() and not config.model_type.is_video_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(patchify_proj=self.audio_patchify_proj, adaln=self.audio_adaln_single, caption_projection=self.audio_caption_projection, inner_dim=self.audio_inner_dim, max_pos=config.audio_positional_embedding_max_pos, num_attention_heads=self.audio_num_attention_heads, use_middle_indices_grid=config.use_middle_indices_grid, timestep_scale_multiplier=config.timestep_scale_multiplier, positional_embedding_theta=config.positional_embedding_theta, rope_type=config.rope_type, double_precision_rope=config.double_precision_rope)

    def _init_transformer_blocks(self, config: LTXModelConfig) -> None:
        video_config = config.get_video_config()
        audio_config = config.get_audio_config()
        self.transformer_blocks = {idx: BasicAVTransformerBlock(idx=idx, video=video_config, audio=audio_config, rope_type=config.rope_type, norm_eps=config.norm_eps) for idx in range(config.num_layers)}

    def _process_transformer_blocks(self, video: Optional[TransformerArgs], audio: Optional[TransformerArgs]) -> Tuple[Optional[TransformerArgs], Optional[TransformerArgs]]:
        for block in self.transformer_blocks.values():
            video, audio = block(video=video, audio=audio)
        return video, audio

    def _process_output(self, scale_shift_table: mx.array, norm_out: nn.LayerNorm, proj_out: nn.Linear, x: mx.array, embedded_timestep: mx.array) -> mx.array:
        table_expanded = scale_shift_table[None, None, :, :]
        timestep_expanded = embedded_timestep[:, :, None, :]
        scale_shift_values = table_expanded + timestep_expanded
        shift = scale_shift_values[:, :, 0, :]
        scale = scale_shift_values[:, :, 1, :]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def __call__(self, video: Optional[Modality] = None, audio: Optional[Modality] = None) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")
        video_args = self.video_args_preprocessor.prepare(video) if video is not None else None
        audio_args = None
        video_out, audio_out = self._process_transformer_blocks(video=video_args, audio=audio_args)
        vx = self._process_output(self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep) if video_out is not None else None
        ax = None
        return vx, ax


class X0Model(nn.Module):
    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def __call__(self, video: Optional[Modality] = None, audio: Optional[Modality] = None) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        vx, ax = self.velocity_model(video, audio)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
