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

class PyTorchExtMinimalMedian(function):

    def __init__(self):
        pass

    def __call__(self, ims):
        return em_pre_torch_ext.median_filter(ims)


class PyTorchExtMedian(function):
    """
    Temporal filtering using em_pre_torch_ext's 3D median filter.
    """
    def __init__(self):
        pass

    def __call__(self, ims):
        window = [len(ims) / 2, 0, 0]
        return em_pre_torch_ext.median_filter(ims, window)[len(ims) / 2]


class NdImageMedian(function):
    """
    Temporal Filtering using scipy.ndimage.median_filter.
    """

    def __init__(self):
        pass

    def __call__(self, ims):
        size = [len(ims), 1, 1]
        return nd.median_filter(ims.numpy(), size=size)
