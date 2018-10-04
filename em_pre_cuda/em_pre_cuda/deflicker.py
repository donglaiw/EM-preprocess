"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * General de-flickering algorithm for EM slices.
 ********************************************************************************************************************"""

import torch


def de_flicker(slices, pre_process, spatial_filter, temporal_filter):
    slices = [pre_process(slc) for slc in slices]
    im_id = len(slices) / 2
    target_im = slices[im_id]
    slices = [spatial_filter(slc) for slc in slices]
    mean_tensor = torch.stack(slices)
    filter_r = temporal_filter(mean_tensor)
    filter_rd = filter_r - mean_tensor[im_id]
    im_diff = spatial_filter(filter_rd)
    return torch.clamp(target_im + im_diff, 0, 255)
