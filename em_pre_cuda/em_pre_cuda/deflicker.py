"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * General de-flickering algorithm for EM slices.
 ********************************************************************************************************************"""

import torch


def de_flicker(slices, spatial_filter, temporal_filter):
    im_id = len(slices) / 2
    mean_tensor = torch.stack([spatial_filter(slc) for slc in slices])
    filter_r = temporal_filter(mean_tensor)
    filter_rd = filter_r - mean_tensor[im_id]
    im_diff = spatial_filter(filter_rd)
    return torch.clamp(slices[im_id] + im_diff, 0, 255)
