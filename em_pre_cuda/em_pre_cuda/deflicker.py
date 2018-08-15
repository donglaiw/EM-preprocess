"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * De-flickering algorithm implemented using Pytorch built-in functions and Pytorch extensions.
 ********************************************************************************************************************"""

import cv2
import torch
import em_pre_torch_ext
from torch.nn.functional import conv2d
from torch.nn.functional import pad
import numpy as np


def _pre_process(ims, global_stat_sz, mask_thres, global_stat=None, global_stat_opt=0):

    if global_stat_opt == 0:
        # mean/std
        if global_stat is not None:
            mm = ims.mean(dim=0).mean(dim=0)
            if global_stat[1] > 0:
                tmp_std = ims.std().item()
                if tmp_std < 1e-3:
                    ims.sub_(mm).add_(global_stat[0])
                else:
                    ims.sub_(mm).div_(tmp_std).mul_(global_stat[1]).add_(global_stat[0])
            else:
                if global_stat[0] > 0:
                    ims.sub_(mm).add_(global_stat[0])

    elif global_stat_opt == 1:
        if global_stat is not None:
            ims_copy = ims[::global_stat_sz, ::global_stat_sz]
            if mask_thres[0] is not None:
                ims_copy = ims_copy[ims_copy > mask_thres[0]]
            if mask_thres[1] is not None:
                ims_copy = ims_copy[ims_copy < mask_thres[1]]
            ims.sub_(ims_copy.median()).add_(global_stat[0])
        if mask_thres[0] is not None:  # for artifact/boundary
            ims[ims < mask_thres[0]] = mask_thres[0]
        if mask_thres[1] is not None:  # for blood vessel
            ims[ims > mask_thres[1]] = mask_thres[1]
    else:
        raise NotImplementedError("The passed global stat op argument (%d) is not implemented." % global_stat_opt)
    return ims


def _spatial_filter(img, kernel, padding):
    unsqz_im = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
    pad_img = pad(unsqz_im, padding, mode='reflect')
    output = conv2d(pad_img, kernel)
    return torch.squeeze(output)


def _temporal_filter(ims, window):
    return em_pre_torch_ext.median_filter(ims, window)


def deflicker_online(get_im, num_slice=100, opts=(0, 0, 0),
                     mask_thres=(10, 245),
                     global_stat_sz=10,
                     global_stat=None,
                     s_flt_rad=15,
                     t_flt_rad=2,
                     verbose=True,
                     write_dir=None):
    # filters: spatial=(filterS_hsz, opts[1]), temporal=(filterT_hsz, opts[2])
    # seq stats: im_size, globalStat, globalStatOpt

    # Verbose printing:
    def _print(content):
        if verbose:
            print(content)

    # Temporal window initialization:
    t_flt_diam = 2 * t_flt_rad + 1
    temporal_filter = torch.tensor([0, 0, t_flt_rad], device='cpu', dtype=torch.float32)

    # Spatial Filter Kernel/Padding Creation:
    spat_padding = (s_flt_rad, ) * 4
    spat_flt_diam = 2 * s_flt_rad + 1
    if opts[1] == 0:  # mean filter: sum to 1
        spatial_kernel = torch.ones((spat_flt_diam, ) * 2, device='cuda').div_(spat_flt_diam ** 2)
        # Un-squeeze the kernel twice for Pytorch's conv2d function.
        spatial_kernel = torch.unsqueeze(torch.unsqueeze(spatial_kernel, 0), 0)
    else:
        raise NotImplementedError('need to implement')

    _print('1. Attempting global normalization...')
    im0 = get_im(0)
    im_size = im0.size()
    if global_stat is None:
        if opts[0] == 0:
            # stat of the first image
            global_stat = [im0.mean(), im0.std()]
        else:
            raise NotImplementedError('Feature with opts: %d needs to be implemented.' % opts[0])
    _print('Global normalization complete.')
    _print('2. Attempting local normalization...')
    mean_tensor = torch.zeros((im_size[0], im_size[1], t_flt_diam), device='cuda')
    # initial chunk
    # e.g. sizeT=7, filterT_hsz=3, mid_r=3
    for i in range(t_flt_rad + 1):  # 0-3
        # flip the initial frames to pad
        mean_tensor[:, :, i] = _spatial_filter(_pre_process(get_im(t_flt_rad - i), global_stat_sz, mask_thres,
                                                            global_stat, opts[0]), spatial_kernel, spat_padding)
    for i in range(t_flt_rad - 1):  # 0-1 -> 4-5
        mean_tensor[:, :, t_flt_rad + 1 + i] = mean_tensor[:, :, t_flt_rad - 1 - i]
    # online change chunk
    im_id = t_flt_rad  # image
    chunk_id = t_flt_diam - 1
    if write_dir is None:
        final_out = np.zeros((im_size[0], im_size[1], num_slice))
    _print('Processing:')
    for i in range(num_slice):
        print '%s/%s' % (i + 1, num_slice)
        # current frame
        im = _pre_process(get_im(i), global_stat, opts[0])

        # last frame needed for temporal filter
        if t_flt_rad + i < num_slice:
            im_m = _pre_process(get_im(t_flt_rad + i), global_stat_sz, mask_thres, global_stat, opts[0])
        else:  # reflection mean
            im_m = _pre_process(get_im(num_slice - 1 - t_flt_rad + (num_slice - 1 - i)), global_stat_sz, mask_thres,
                                global_stat, opts[0])
        mean_tensor[:, :, chunk_id] = _spatial_filter(im_m, spatial_kernel, spat_padding)

        # local temporal filtering
        if opts[2] == 0:  # median filter
            filter_r = _temporal_filter(mean_tensor, temporal_filter)
        else:
            raise NotImplementedError('need to implement')

        filter_rd = filter_r[:, :, t_flt_rad] - mean_tensor[:, :, im_id]
        im_diff = _spatial_filter(filter_rd, spatial_kernel, spat_padding)
        out_im = torch.clamp(im + im_diff, 0, 255).cpu().numpy()
        if write_dir is None:
            final_out[:, :, i] = out_im
        else:
            cv2.imwrite(write_dir % (i + 1), out_im)
        im_id = (im_id + 1) % t_flt_diam
        chunk_id = (chunk_id + 1) % t_flt_diam
    _print("Local normalization complete.")

    if write_dir is None:
        return final_out
