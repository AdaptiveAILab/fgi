import torch

import fgi
from .. import functional

################################################################
# Neuron update functional
################################################################

DEFAULT_ALIF_ALPHA = 0.9
DEFAULT_ALIF_ADAPTIVE_ALPHA_MEAN = 3.0
DEFAULT_ALIF_ADAPTIVE_ALPHA_STD = 0.1

DEFAULT_ALIF_RHO = 0.9
DEFAULT_ALIF_ADAPTIVE_RHO_MEAN = 4.0
DEFAULT_ALIF_ADAPTIVE_RHO_STD = 0.5
DEFAULT_ALIF_THETA = 0.1
DEFAULT_ALIF_BETA = 0.8


def alif_update(
        x: torch.Tensor,
        u: torch.Tensor,
        a: torch.Tensor,
        alpha: torch.Tensor,
        beta: float,
        rho: torch.Tensor,
        theta: float = DEFAULT_ALIF_THETA
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # determine dynamic threshold.
    theta_t = theta + a.mul(beta)

    # membrane potential update
    u = u.mul(alpha) + x.mul(1.0 - alpha)

    # generate spike.
    z = functional.StepGaussianGrad.apply(u - theta_t)

    # reset membrane potential.
    # soft reset (keeps remaining membrane potential)
    u = u - z.mul(theta_t)

    # adapt spike accumulator.
    # a = a.mul(rho) + z
    # Variant from
    a = a.mul(rho) + z.mul(1.0 - rho)
    return z, u, a


def alif_fgi_update(
        x: torch.Tensor,
        u: torch.Tensor,
        a: torch.Tensor,
        alpha: torch.Tensor,
        beta: float,
        rho: torch.Tensor,
        theta: float = DEFAULT_ALIF_THETA
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # determine dynamic threshold.
    theta_t = theta + a.mul(beta)

    # membrane potential update
    u = u.mul(alpha) + x.mul(1.0 - alpha)

    # generate spike.
    utheta = u - theta_t
    z = fgi.fgi(utheta, functional.step(utheta), functional.gaussian(utheta))

    # reset membrane potential.
    # soft reset (keeps remaining membrane potential)
    u = u - z.mul(theta_t)

    # adapt spike accumulator.
    # a = a.mul(rho) + z
    # Variant from
    a = a.mul(rho) + z.mul(1.0 - rho)
    return z, u, a

################################################################
# Layer classes
################################################################

class ALIFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            alpha: float = DEFAULT_ALIF_ALPHA,
            adaptive_alpha: bool = True,
            adaptive_alpha_mean: float = DEFAULT_ALIF_ADAPTIVE_ALPHA_MEAN,
            adaptive_alpha_std: float = DEFAULT_ALIF_ADAPTIVE_ALPHA_STD,
            beta: float = DEFAULT_ALIF_BETA,
            rho: float = DEFAULT_ALIF_RHO,
            adaptive_rho: bool = True,
            adaptive_rho_mean: float = DEFAULT_ALIF_ADAPTIVE_RHO_MEAN,
            adaptive_rho_std: float = DEFAULT_ALIF_ADAPTIVE_RHO_STD,
            bias: bool = False
    ) -> None:
        super(ALIFCell, self).__init__()

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

        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_mean = adaptive_rho_mean
        self.adaptive_rho_std = adaptive_rho_std

        rho = rho * torch.ones(layer_size)

        if adaptive_rho:
            self.rho = torch.nn.Parameter(rho)
            torch.nn.init.normal_(self.rho, mean=adaptive_rho_mean, std=adaptive_rho_std)
        else:
            self.register_buffer("rho", rho)

        self.beta = beta

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, a = state

        in_sum = self.linear(x)

        alpha = self.alpha

        if self.adaptive_alpha:
            alpha = torch.sigmoid(self.alpha)

        rho = self.rho

        if self.adaptive_rho:
            rho = torch.sigmoid(self.rho)

        z, u, a = alif_update(
            x=in_sum,
            u=u,
            a=a,
            alpha=alpha,
            beta=self.beta,
            rho=rho,
        )

        return z, u, a


class ALIFFGICell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            alpha: float = DEFAULT_ALIF_ALPHA,
            adaptive_alpha: bool = True,
            adaptive_alpha_mean: float = DEFAULT_ALIF_ADAPTIVE_ALPHA_MEAN,
            adaptive_alpha_std: float = DEFAULT_ALIF_ADAPTIVE_ALPHA_STD,
            beta: float = DEFAULT_ALIF_BETA,
            rho: float = DEFAULT_ALIF_RHO,
            adaptive_rho: bool = True,
            adaptive_rho_mean: float = DEFAULT_ALIF_ADAPTIVE_RHO_MEAN,
            adaptive_rho_std: float = DEFAULT_ALIF_ADAPTIVE_RHO_STD,
            bias: bool = False
    ) -> None:
        super(ALIFFGICell, self).__init__()

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

        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_mean = adaptive_rho_mean
        self.adaptive_rho_std = adaptive_rho_std

        rho = rho * torch.ones(layer_size)

        if adaptive_rho:
            self.rho = torch.nn.Parameter(rho)
            torch.nn.init.normal_(self.rho, mean=adaptive_rho_mean, std=adaptive_rho_std)
        else:
            self.register_buffer("rho", rho)

        self.beta = beta

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, a = state

        in_sum = self.linear(x)

        alpha = self.alpha

        if self.adaptive_alpha:
            alpha = torch.sigmoid(self.alpha)

        rho = self.rho

        if self.adaptive_rho:
            rho = torch.sigmoid(self.rho)

        z, u, a = alif_fgi_update(
            x=in_sum,
            u=u,
            a=a,
            alpha=alpha,
            beta=self.beta,
            rho=rho,
        )

        return z, u, a