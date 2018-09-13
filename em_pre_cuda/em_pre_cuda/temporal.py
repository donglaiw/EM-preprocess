"""*********************************************************************************************************************
 * Name: temporal.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * Specification of temporal filtering for the deflicker algorithm.
 ********************************************************************************************************************"""
import torch
import em_pre_torch_ext
from abc import ABCMeta, abstractmethod
from scipy import ndimage as nd


class TemporalFilter:
    """
    Base class for any temporal filtering used in the de-flickering algorithm. The __init__ method is used to pass any
    necessary argument and the __call__ method makes it easy to call this as a function on a 3D PyTorch Tensor.
    Usually, the window used by the concrete classes is [1, 1, len(ims)].
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, ims):
        """
        Calls the temporal filtering algorithm on a 3D PyTorch Tensor.
        :param ims: The stack of EM-slices as a 3D PyTorch Tensor.
        :return: The middle slice off of the result of temporal filtering.
        """
        pass


class PyTorchExtMinimalMedian(TemporalFilter):

    def __init__(self):
        pass

    def __call__(self, ims):
        return em_pre_torch_ext.median_filter(ims)


class PyTorchExtMedian(TemporalFilter):
    """
    Temporal filtering using em_pre_torch_ext's 3D median filter.
    """
    def __init__(self):
        pass

    def __call__(self, ims):
        window = torch.tensor([0, 0, len(ims) / 2], dtype=torch.float32)
        return em_pre_torch_ext.median_filter(ims, window)[len(ims) / 2]


class NdImageMedian(TemporalFilter):
    """
    Temporal Filtering using scipy.ndimage.median_filter.
    """

    def __init__(self):
        pass

    def __call__(self, ims):
        size = [1, 1, len(ims)]
        return nd.median_filter(ims.numpy(), size=size)
