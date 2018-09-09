

import torch
import em_pre_torch_ext


class PyTorchExtMinimalMedian:
    def __init__(self):
        pass

    def __call__(self, ims):
        return em_pre_torch_ext.median_filter(ims)


class PyTorchExtMedian:
    def __init__(self, rad_z):
        self.window = torch.tensor([0, 0, rad_z], device='cpu', dtype=torch.float32)

    def __call__(self, ims):
        return em_pre_torch_ext.median_filter(ims, self.window)
