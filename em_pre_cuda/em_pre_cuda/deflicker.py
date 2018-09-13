"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * General de-flickering algorithm for EM slices.
 ********************************************************************************************************************"""

import torch


def deflicker(slice_getter, spatial_filter, temporal_filter, slice_range):
    """
    General de-flickering algorithm using PyTorch.
    :param slice_getter: A callable that provides the algorithm with a globally pre-processed slice. See
    :param spatial_filter: A callable that applies spatial filtering to a single slice.
    :param temporal_filter: A callable that applies temporal filtering to a stack of slices and returns the middle
    slice.
    :param slice_range: The range of slices used to deflicker the middle slice.
    :return: The de-flickered slice in the middle of slice_range.
    """
    mean_tensor = torch.stack([spatial_filter(slice_getter[i]) for i in slice_range])
    im_id = len(slice_range) / 2
    target_idx = slice_range[im_id]
    filter_r = temporal_filter(mean_tensor)
    filter_rd = filter_r - mean_tensor[im_id]
    im_diff = spatial_filter(filter_rd)
    return torch.clamp(slice_getter[target_idx] + im_diff, 0, 255)
