import mlx.core as mx
import mlx.nn as nn


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, bias: bool = True):
        super().__init__()
        out_features = out_features or hidden_size
        self.linear1 = nn.Linear(in_features, hidden_size, bias=bias)
        self.act = nn.GELU(approx="tanh")
        self.linear2 = nn.Linear(hidden_size, out_features, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x
