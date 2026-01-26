import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"
    TWO_D = "2d"


class AttentionType(Enum):
    DEFAULT = "default"


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "BaseModelConfig":
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

    def to_dict(self) -> dict[str, Any]:
        result = {}
        for k, v in self.__dict__.items():
            if v is not None:
                if isinstance(v, Enum):
                    result[k] = v.value
                elif hasattr(v, 'to_dict'):
                    result[k] = v.to_dict()
                else:
                    result[k] = v
        return result


@dataclass
class TransformerConfig(BaseModelConfig):
    dim: int
    heads: int
    d_head: int
    context_dim: int


@dataclass
class VideoVAEConfig(BaseModelConfig):
    convolution_dimensions: int = 3
    in_channels: int = 3
    out_channels: int = 128
    latent_channels: int = 128
    patch_size: int = 4
    encoder_blocks: List[tuple] = field(default_factory=lambda: [
        ("res_x", {"num_layers": 4}),
        ("compress_space_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_time_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 6}),
        ("compress_all_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
        ("compress_all_res", {"multiplier": 2}),
        ("res_x", {"num_layers": 2}),
    ])
    decoder_blocks: List[tuple] = field(default_factory=lambda: [
        ("res_x", {"num_layers": 5, "inject_noise": False}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5, "inject_noise": False}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5, "inject_noise": False}),
        ("compress_all", {"residual": True, "multiplier": 2}),
        ("res_x", {"num_layers": 5, "inject_noise": False}),
    ])


@dataclass
class LTXModelConfig(BaseModelConfig):
    model_type: LTXModelType = LTXModelType.AudioVideo
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    in_channels: int = 128
    out_channels: int = 128
    num_layers: int = 48
    cross_attention_dim: int = 4096
    caption_channels: int = 3840
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_cross_attention_dim: int = 2048
    audio_caption_channels: int = 3840
    positional_embedding_theta: float = 10000.0
    positional_embedding_max_pos: Optional[List[int]] = None
    audio_positional_embedding_max_pos: Optional[List[int]] = None
    use_middle_indices_grid: bool = True
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED
    double_precision_rope: bool = False
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1000
    norm_eps: float = 1e-6
    attention_type: AttentionType = AttentionType.DEFAULT
    vae_config: Optional[VideoVAEConfig] = None

    def __post_init__(self):
        if self.positional_embedding_max_pos is None:
            self.positional_embedding_max_pos = [20, 2048, 2048]
        if self.audio_positional_embedding_max_pos is None:
            self.audio_positional_embedding_max_pos = [20]
        if isinstance(self.model_type, str):
            self.model_type = LTXModelType(self.model_type)
        if isinstance(self.rope_type, str):
            self.rope_type = LTXRopeType(self.rope_type)
        if isinstance(self.attention_type, str):
            self.attention_type = AttentionType(self.attention_type)

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim

    def get_video_config(self) -> Optional[TransformerConfig]:
        if not self.model_type.is_video_enabled():
            return None
        return TransformerConfig(
            dim=self.inner_dim,
            heads=self.num_attention_heads,
            d_head=self.attention_head_dim,
            context_dim=self.cross_attention_dim,
        )

    def get_audio_config(self) -> Optional[TransformerConfig]:
        if not self.model_type.is_audio_enabled():
            return None
        return TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.audio_num_attention_heads,
            d_head=self.audio_attention_head_dim,
            context_dim=self.audio_cross_attention_dim,
        )
