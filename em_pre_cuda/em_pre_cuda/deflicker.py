"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * De-flickering algorithm implemented using Pytorch built-in functions and Pytorch extensions.
 ********************************************************************************************************************"""

import cv2
import torch
import em_pre_torch_ext
from torch.nn.functional import conv2d, pad
import numpy as np


def _pre_process(image, global_stat, method='naive', sampling_step=10, mask_thres=(10, 245)):
    """TODO
    :param image: the desired image as a Pytorch Tensor.
    :param global_stat: A two element tuple containing info regarding the desired global statistic of the image stack,
    the first element being the mean and the second being the std.
    :param method: the desired method of pre-processing:<ol><li>naive</li><li>threshold</li></ol>.
    :param sampling_step: step size used to sample the image for the threshold method.
    :param mask_thres: A two element tuple containing both minimum and maximum pixel threshold values for the threshold
    method.
    :return: the pre-processed image.
    """

    if method == 'naive':
        # mean/std
        if global_stat is not None:
            mm = image.mean(dim=0).mean(dim=0)
            if global_stat[1] > 0:
                tmp_std = image.std().item()
                if tmp_std < 1e-3:
                    image.sub_(mm).add_(global_stat[0])
                else:
                    image.sub_(mm).div_(tmp_std).mul_(global_stat[1]).add_(global_stat[0])
            else:
                if global_stat[0] > 0:
                    image.sub_(mm).add_(global_stat[0])

    elif method == 'threshold':
        if global_stat is not None:
            im_copy = image[::sampling_step, ::sampling_step]
            if mask_thres[0] is not None:
                im_copy = im_copy[im_copy > mask_thres[0]]
            if mask_thres[1] is not None:
                im_copy = im_copy[im_copy < mask_thres[1]]
            image.sub_(im_copy.median()).add_(global_stat[0])
        if mask_thres[0] is not None:  # for artifact/boundary
            image[image < mask_thres[0]] = mask_thres[0]
        if mask_thres[1] is not None:  # for blood vessel
            image[image > mask_thres[1]] = mask_thres[1]
    else:
        raise NotImplementedError("The passed global stat op argument (%s) is not implemented." % method)
    return image


def _spatial_filter(img, kernel, padding):
    unsqz_im = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
    pad_img = pad(unsqz_im, padding, mode='reflect')
    output = conv2d(pad_img, kernel)
    return torch.squeeze(output)


def _temporal_filter(ims, window, method):
    if method == 'median':
        return em_pre_torch_ext.median_filter(ims, window)
    else:
        raise NotImplementedError("The passed global stat op argument (%s) is not implemented." % method)


def _create_t_flt_window(method, radius):
    if method == 'median':
        return torch.tensor([0, 0, radius], device='cpu', dtype=torch.float32)
    else:
        raise NotImplementedError("The passed global stat op argument (%s) is not implemented." % method)


def _create_spatial_kernel(method, diam):
    if method == 'mean':  # mean filter: sum to 1
        spatial_kernel = torch.ones((diam,) * 2, device='cuda').div_(diam ** 2)
        # Un-squeeze the kernel twice for Pytorch's conv2d function.
        return torch.unsqueeze(torch.unsqueeze(spatial_kernel, 0), 0)
    else:
        raise NotImplementedError("The passed global stat op argument (%s) is not implemented." % method)


def deflicker_online(get_im, slice_range=None, global_stat=None,
                     pre_proc_method='naive', sampling_step=10, mask_thres=(10, 245),
                     spat_flt_method='mean', s_flt_rad=15,
                     temp_flt_method='median', t_flt_rad=2,
                     verbose=True,
                     write_dir=None):
    """
    A sequential implementation of the de-flickering algorithm on CUDA.
    :param get_im: Image getter function. Should take the desired index as argument and return its corresponding image
    as a Pytorch CUDA Tensor.
    :param slice_range: Number of image slices to be de-flickered.
    :param global_stat: The desired global statistics of the image stack. Is a two element tuple in the form of
    (global_mean, global_std).
    :param pre_proc_method: The method of pre-processing to be applied to individual images. See _pre_process for more
    info on each option.
    :param sampling_step: Refer to _pre_process.
    :param mask_thres: Refer to _pre_process.
    :param spat_flt_method:
    :param s_flt_rad: The radius of the spatial filter.
    :param temp_flt_method: The method of temporal filtering applied.
    :param t_flt_rad: The radius of the temporal filter.
    :param verbose: Verbosity level.
    :param write_dir: Whether to save individual images to disk straight away after de-flickering after each iteration.
    :return: The de-flickered image as a Pytorch CPU Tensor if write_dir is None. Else, None.
    """

    # filters: spatial=(filterS_hsz, opts[1]), temporal=(filterT_hsz, opts[2])
    # seq stats: im_size, globalStat, globalStatOpt

    # Verbose printing:
    if slice_range is None:
        slice_range = range(0, 100)

    def _print(content):
        if verbose:
            print(content)

    # Temporal window initialization:
    t_flt_diam = 2 * t_flt_rad + 1
    temp_flt_window = _create_t_flt_window(temp_flt_method, t_flt_rad)

    # Spatial Filter Kernel/Padding Creation:
    spat_padding = (s_flt_rad,) * 4
    spat_flt_diam = 2 * s_flt_rad + 1
    spatial_kernel = _create_spatial_kernel(spat_flt_method, spat_flt_diam)

    _print('1. Starting global normalization...')
    im0 = get_im(0)
    im_size = im0.size()
    if global_stat is None:
        if pre_proc_method == 'naive':
            # stat of the first image
            global_stat = [im0.mean(), im0.std()]
        else:
            raise NotImplementedError('Feature with opts: %d needs to be implemented.' % pre_proc_method)
    _print('Global normalization complete.')
    _print('2. Starting local normalization...')
    mean_tensor = torch.zeros((im_size[0], im_size[1], t_flt_diam), device='cuda')
    # initial chunk
    # e.g. sizeT=7, filterT_hsz=3, mid_r=3
    for i in range(t_flt_rad + 1):  # 0-3
        # flip the initial frames to pad
        mean_tensor[:, :, i] = _spatial_filter(
            _pre_process(get_im(t_flt_rad - i), global_stat, pre_proc_method, sampling_step, mask_thres),
            spatial_kernel, spat_padding)
    for i in range(t_flt_rad - 1):  # 0-1 -> 4-5
        mean_tensor[:, :, t_flt_rad + 1 + i] = mean_tensor[:, :, t_flt_rad - 1 - i]
    # online change chunk
    im_id = t_flt_rad  # image
    chunk_id = t_flt_diam - 1
    num_slices = len(slice_range)
    if write_dir is None:
        final_out = np.zeros((im_size[0], im_size[1], num_slices))
    print('Processing:')

    for i in slice_range:
        print('%s/%s' % (i + 1, num_slices))
        # current frame
        im = _pre_process(get_im(i), global_stat, pre_proc_method, sampling_step, mask_thres)

        # last frame needed for temporal filter
        if t_flt_rad + i < num_slices:
            im_m = _pre_process(get_im(t_flt_rad + i), global_stat, pre_proc_method, sampling_step, mask_thres)
        else:  # reflection mean
            im_m = _pre_process(get_im(num_slices - 1 - t_flt_rad + (num_slices - 1 - i)), global_stat,
                                pre_proc_method,
                                sampling_step, mask_thres)
        mean_tensor[:, :, chunk_id] = _spatial_filter(im_m, spatial_kernel, spat_padding)
        del im_m
        # local temporal filtering
        filter_r = _temporal_filter(mean_tensor, temp_flt_window, temp_flt_method)

        filter_rd = filter_r[:, :, t_flt_rad] - mean_tensor[:, :, im_id]
        del filter_r
        im_diff = _spatial_filter(filter_rd, spatial_kernel, spat_padding)
        del filter_rd
        out_im = torch.clamp(im + im_diff, 0, 255).cpu().numpy()
        del im_diff, im
        if write_dir is None:
            final_out[:, :, i] = out_im
        else:
            cv2.imwrite(write_dir % (i + 1), out_im)
        del out_im
        im_id = (im_id + 1) % t_flt_diam
        chunk_id = (chunk_id + 1) % t_flt_diam
    _print("Local normalization complete.")

    if write_dir is None:
        return final_out
