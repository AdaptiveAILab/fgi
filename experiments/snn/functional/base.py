import torch

################################################################
# Simple base functional
################################################################

def step(x: torch.Tensor) -> torch.Tensor:
    #
    # x.gt(0.0).float()
    # is slightly faster (but less readable) than
    # torch.where(x > 0.0, 1.0, 0.0)
    #
    return x.gt(0.0).float()

def exp_decay(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.abs(x))

def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )

def std_gaussian(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(
        -0.5 * (x ** 2)
    )

def linear_peak(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - torch.abs(x))