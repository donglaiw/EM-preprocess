"""*********************************************************************************************************************
 * Name: deflicker.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * De-flickering algorithms implemented using Pytorch extensions.
 ********************************************************************************************************************"""

import cv2
import torch
import em_pre_torch_ext
from torch.nn.functional import conv2d
from torch.nn import Conv2d, init
from torch.nn.functional import pad
import numpy as np


# TODO: Ask Donglai to write a better doc for this.

def _pre_process(ims, global_stat=None, global_stat_opt=0):
    """
    Pre-processes the passed image stack based on the parameters passed to it.
    :param ims: the stack of images as a Pytorch Tensor. Assumes the sizes represent x, y and z respectively with z
    being the image index.
    :param global_stat:
    :param global_stat_opt:
    :return: the preprocessed stack of images as a Pytorch Tensor.
    """
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
    else:
        raise NotImplementedError("The passed global stat op argument (%d) is not implemented." % global_stat_opt)
    return ims


def _create_filters(filter_s_hsz, filter_t_hsz, opts):
    spat_size = [x * 2 + 1 for x in filter_s_hsz]
    temp_size = 2 * filter_t_hsz + 1
    if opts[1] == 0:  # mean filter: sum to 1
        spat_filter = torch.ones(spat_size, device='cuda').div_(spat_size[0] * spat_size[1])
    else:
        raise NotImplementedError('need to implement')
    return spat_filter, temp_size


def _conv2d(img, flt_shape, flt_rad):
    unsqz_im = torch.unsqueeze(torch.unsqueeze(img, 0), 0)
    # print "filter shape:"
    pad_t = (flt_rad[0], flt_rad[1], flt_rad[0], flt_rad[1])
    unsqz_flt = torch.unsqueeze(torch.unsqueeze(flt_shape, 0), 0)
    pad_img = pad(unsqz_im, pad_t, mode='reflect')
    output = conv2d(pad_img, unsqz_flt)
    # m = Conv2d(1, 1, kernel_size=(31, 31)).cuda()
    # weight = torch.tensor(1. / (flt_shape[0] * flt_shape[1]), device='cuda', dtype=torch.float32)
    # print weight
    # init.constant_(m.weight,  weight)
    # output = torch.squeeze(m(pad_img))
    # print output.size()
    # return output
    return torch.squeeze(output)


def _3d_median_filter(ims, filter_shape):
    flt = torch.tensor(filter_shape, device='cpu', dtype=torch.float32) / 2
    return em_pre_torch_ext.median_filter(ims, flt)


def deflicker_online(get_im, num_slice=100, opts=(0, 0, 0),
                     global_stat=None, s_flt_rad=(15, 15), t_flt_rad=2):
    """
    :param get_im:
    :param num_slice:
    :param opts:
    :param global_stat:
    :param s_flt_rad:
    :param t_flt_rad:
    :return:
    """
    # filters: spatial=(filterS_hsz, opts[1]), temporal=(filterT_hsz, opts[2])
    # seq stats: im_size, globalStat, globalStatOpt
    im0 = get_im(0)
    im_size = im0.size()
    spatial_filter, temp_size = _create_filters(s_flt_rad, t_flt_rad, opts)
    print('1. Attempting global normalization...')
    if global_stat is None:
        if opts[0] == 0:
            # stat of the first image
            global_stat = [im0.mean(), im0.std()]
        else:
            raise NotImplementedError('Feature with opts: %d needs to be implemented.' % opts[0])
    print('Global normalization complete.')
    print('2. Attempting local normalization...')
    mean_tensor = torch.zeros((im_size[0], im_size[1], temp_size), device='cuda')
    # initial chunk
    # e.g. sizeT=7, filterT_hsz=3, mid_r=3
    for i in range(t_flt_rad + 1):  # 0-3
        # flip the initial frames to pad
        mean_tensor[:, :, i] = _conv2d(_pre_process(get_im(t_flt_rad - i), global_stat, opts[0]),
                                       spatial_filter, s_flt_rad)
    for i in range(t_flt_rad - 1):  # 0-1 -> 4-5
        mean_tensor[:, :, t_flt_rad + 1 + i] = mean_tensor[:, :, t_flt_rad - 1 - i]
    # online change chunk
    im_id = t_flt_rad  # image
    chunk_id = temp_size - 1
    final_out = np.zeros((im_size[0], im_size[1], num_slice))
    for i in range(num_slice):
        print 'Processing: %s/%s' % (i + 1, num_slice)
        # current frame
        im = _pre_process(get_im(i), global_stat, opts[0])

        # last frame needed for temporal filter
        if t_flt_rad + i < num_slice:
            imM = _pre_process(get_im(t_flt_rad + i), global_stat, opts[0])
        else:  # reflection mean
            imM = _pre_process(get_im(num_slice - 1 - t_flt_rad + (num_slice - 1 - i)), global_stat, opts[0])
        mean_tensor[:, :, chunk_id] = _conv2d(imM, spatial_filter, s_flt_rad)

        # local temporal filtering
        if opts[2] == 0:  # median filter
            filterR = _3d_median_filter(mean_tensor, [1, 1, temp_size])
        else:
            raise NotImplementedError('need to implement')

        filterRD = filterR[:, :, t_flt_rad] - mean_tensor[:, :, im_id]
        imDiff = _conv2d(filterRD, spatial_filter, s_flt_rad)
        out_im = torch.clamp(im + imDiff, 0, 255).cpu().numpy()
        cv2.imwrite("output_gpu_%d.png" % (i + 1), out_im)
        final_out[:, :, i] = out_im
        im_id = (im_id + 1) % temp_size
        chunk_id = (chunk_id + 1) % temp_size
