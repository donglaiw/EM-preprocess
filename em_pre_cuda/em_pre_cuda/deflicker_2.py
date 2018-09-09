"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * De-flickering algorithm implemented using Pytorch built-in functions and Pytorch extensions.
 ********************************************************************************************************************"""

import torch


def deflicker(slice_getter, spatial_filter, temporal_filter, slice_range):
    mean_tensor = torch.stack([spatial_filter(slice_getter[i]) for i in slice_range])
    im_id = len(slice_range) / 2
    target_idx = slice_range(im_id)
    filter_r = temporal_filter(mean_tensor)
    filter_rd = filter_r - mean_tensor[:, :, im_id]
    im_diff = spatial_filter(filter_rd)
    return torch.clamp(slice_getter[target_idx] + im_diff, 0, 255)
