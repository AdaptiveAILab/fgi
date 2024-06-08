import torch
from .. import functional

################################################################
# Neuron update functional
################################################################

DEFAULT_LI_ALPHA = 0.9
DEFAULT_LI_ADAPTIVE_ALPHA_MEAN = 3.0
DEFAULT_LI_ADAPTIVE_ALPHA_STD = 0.1

DEFAULT_LIF_ALPHA = 0.9
DEFAULT_LIF_ADAPTIVE_ALPHA_MEAN = 1.0
DEFAULT_LIF_ADAPTIVE_ALPHA_STD = 1.0

def li_update(
        x: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor
) -> torch.Tensor:

    u = u.mul(alpha) + x.mul(1.0 - alpha)
    return u

def lif_update(
        x: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor,
        theta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:

    # membrane potential update
    u = u.mul(alpha) + x.mul(1.0 - alpha)

    # generate spike.
    z = functional.StepLinearGrad.apply(u - theta / theta)

    # reset membrane potential.
    # soft reset (keeps remaining membrane potential)
    u = u - z.mul(theta)

    return z, u


################################################################
# Layer classes
################################################################

class LICell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            alpha: float = DEFAULT_LI_ALPHA,
            adaptive_alpha: bool = False,
            adaptive_alpha_mean: float = DEFAULT_LI_ADAPTIVE_ALPHA_MEAN,
            adaptive_alpha_std: float = DEFAULT_LI_ADAPTIVE_ALPHA_STD,
            bias: bool = False
    ) -> None:
        super(LICell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        self.adaptive_alpha = adaptive_alpha
        self.adaptive_alpha_mean = adaptive_alpha_mean
        self.adaptive_alpha_std = adaptive_alpha_std

        alpha = alpha * torch.ones(layer_size)

        if adaptive_alpha:
            self.alpha = torch.nn.Parameter(alpha)
            torch.nn.init.normal_(self.alpha, mean=adaptive_alpha_mean, std=adaptive_alpha_std)
        else:
            self.register_buffer("alpha", alpha)

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        in_sum = self.linear(x)

        alpha = self.alpha

        if self.adaptive_alpha:
            alpha = torch.sigmoid(self.alpha)

        u = li_update(x=in_sum, u=u, alpha=alpha)

        return u


class LIFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            alpha: float = DEFAULT_LIF_ALPHA,
            adaptive_alpha: bool = False,
            adaptive_alpha_mean: float = DEFAULT_LIF_ADAPTIVE_ALPHA_MEAN,
            adaptive_alpha_std: float = DEFAULT_LIF_ADAPTIVE_ALPHA_STD,
            bias: bool = False
    ) -> None:
        super(LIFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        self.adaptive_alpha = adaptive_alpha
        self.adaptive_alpha_mean = adaptive_alpha_mean
        self.adaptive_alpha_std = adaptive_alpha_std

        alpha = alpha * torch.ones(layer_size)

        if adaptive_alpha:
            self.alpha = torch.nn.Parameter(alpha)
            torch.nn.init.normal_(self.alpha, mean=adaptive_alpha_mean, std=adaptive_alpha_std)
        else:
            self.register_buffer("alpha", tensor=alpha)

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        z, u = state
        in_sum = self.linear(x)
        alpha = self.alpha

        if self.adaptive_alpha:
            # Here sigmoid (with quasi linear slope = 1) is used to
            # ensure the range [0, 1] for alpha.
            alpha = torch.sigmoid(2.0 * self.alpha)

        z, u = lif_update(x=in_sum, u=u, alpha=alpha)

        return z, u