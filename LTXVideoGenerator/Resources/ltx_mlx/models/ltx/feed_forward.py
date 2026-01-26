import mlx.core as mx
import mlx.nn as nn


class GELU(nn.Module):
    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    def __call__(self, x: mx.array) -> mx.array:
        if self.approximate == "tanh":
            return nn.gelu_approx(x)
        else:
            return nn.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4, bias: bool = True):
        super().__init__()
        dim_out = dim_out or dim
        inner_dim = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner_dim, bias=bias)
        self.act = GELU(approximate="tanh")
        self.proj_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x
