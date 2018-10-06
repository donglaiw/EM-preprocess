"""*********************************************************************************************************************
 * Name: temporal.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * Specification of temporal filtering for the de-flicker algorithm. These object are callable on a stack of 2D tensors
 with a size of (z, x, y)
 ********************************************************************************************************************"""
import em_pre_torch_ext
from scipy import ndimage as nd
import torch

class PyTorchExtMinimalMedian:
    def __init__(self):
        pass

    def __call__(self, ims):
        return em_pre_torch_ext.median_filter(ims)


class PyTorchExtMedian:
    """
    Temporal filtering using em_pre_torch_ext's 3D median filter.
    """
    def __init__(self):
        pass

    def __call__(self, ims):
        window = [len(ims) / 2, 0, 0]
        return em_pre_torch_ext.median_filter(ims, window)[len(ims) / 2]


class NdImageMedian:
    """
    Temporal Filtering using scipy.ndimage.median_filter.
    """

    def __init__(self):
        pass

    def __call__(self, ims):
        size = [len(ims), 1, 1]
        return torch.from_numpy(nd.median_filter(ims.numpy(), size=size)[len(ims) / 2])
