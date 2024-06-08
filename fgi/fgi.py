import torch

def bypass(
    x: torch.Tensor,
    fx: torch.Tensor
) :
    """
    This function bypasses the gradient of a tensor with the
    gradient of another tensor. Note that the final gradient
    will be determined via autograd, thus gxc is given in form
    of the "antiderivate" of the resulting final derivative.
    """
    assert x.size() == fx.size()
    return fx - fx.detach() + x.detach()


def fgi(x: torch.Tensor, fx: torch.Tensor, fdx: torch.Tensor):
    """
    This function realizes forward gradient injection. In forward
    direction it returns fx. In backward direction the gradient
    received by x is shaped by fdx.

    """
    mul = x * fdx.detach()
    return mul - mul.detach() + fx.detach()