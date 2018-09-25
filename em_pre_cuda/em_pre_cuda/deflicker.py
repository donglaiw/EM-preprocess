"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * General de-flickering algorithm for EM slices.
 ********************************************************************************************************************"""

import torch

# TODO: Implement in C++ maybe?


def de_flicker(slices, spatial_filter, temporal_filter):
    num_slices = slices.size(0)
    im_id = num_slices / 2
    mean_tensor = slices
    filter_r = temporal_filter(mean_tensor)
    filter_rd = filter_r - mean_tensor[im_id]
    im_diff = spatial_filter(filter_rd)
    return torch.clamp(slices[im_id] + im_diff, 0, 255)
