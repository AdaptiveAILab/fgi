import torch

class CUDATimer:

    def __init__(self):
        self._starter = torch.cuda.Event(enable_timing=True)
        self._ender = torch.cuda.Event(enable_timing=True)

    def reset(self):
        self._starter.record()

    def time(self):
        self._ender.record()
        torch.cuda.synchronize()
        forward_time = self._starter.elapsed_time(self._ender)
        return forward_time







