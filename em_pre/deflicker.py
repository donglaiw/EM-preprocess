import cv2
import numpy as np
import scipy.ndimage as nd
import torch
from torch.nn.functional import conv2d
import cuda_3d_median as cuda

def _pre_process_ims(ims, global_stat=None, global_stat_opt=0):
    # im: x,y,t
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
    return ims

def _create_filters(filter_s_hsz, filter_t_hsz, opts, device='cuda'):
    spat_size = [x * 2 + 1 for x in filter_s_hsz]
    temp_size = 2 * filter_t_hsz + 1
    if opts[1] == 0:  # mean filter: sum to 1
        spat_filter = torch.ones(spat_size, device=device).div_(spat_size[0] * spat_size[1])
    else:
        raise Exception('need to implement')
    return spat_filter, temp_size

def de_flicker_batch(ims, opts=(0, 0, 0), globalStat=None, filterS_hsz=(15, 15), filterT_hsz=2):
    # ims: x,y,t
    # opts: globalStat, fliterS, filterT
    # prepare parameters
    sizeS = [x * 2 + 1 for x in filterS_hsz]
    sizeT = 2 * filterT_hsz + 1
    numSlice = ims.shape[2]

    print '1. global normalization'
    if globalStat is None:
        if opts[0] == 0:
            # stat of the first image
            globalStat = [np.mean(ims[:, :, 0]), np.std(ims[:, :, 0])]
        else:
            raise ('need to implement')
    ims = _pre_process_ims(ims.astype(np.float32), globalStat, opts[0])

    print '2. local normalization'
    print '2.1 compute spatial mean'
    if opts[1] == 0:
        filterS = np.ones((sizeS[0], sizeS[1], 1), dtype=np.float32) / (sizeS[0] * sizeS[1])
        meanTensor = nd.convolve(ims, filterS)
    else:
        raise ('need to implement')

    print '2.2 compute temporal median'
    if opts[2] == 0:
        meanTensorF = nd.median_filter(meanTensor, (1, 1, sizeT))
    else:
        raise ('need to implement')
    print '2.3. add back filtered difference'
    out = ims + nd.convolve(meanTensorF - meanTensor, filterS)
    out = np.clip(out, 0, 255).astype(np.uint8)  # (-1).uint8=254
    return out

def _3d_median_filter(ims, filter_shape, device):
    if device == 'cuda':
        output = torch.zeros_like(ims)
        cuda.median_filter(ims, output, *(map(int, ims.size()) + filter_shape))
    else:
        return nd.median_filter(ims.numpy(), filter_shape)

def de_flicker_online(get_im, num_slice=100, opts=(0, 0, 0),
                      global_stat=None, filter_s_hsz=(15, 15), filter_t_hsz=2, device='cuda'):
    # im: x,y,t
    # filters: spatial=(filterS_hsz, opts[1]), temporal=(filterT_hsz, opts[2])
    # seq stats: imSize, globalStat, globalStatOpt
    im0 = get_im(0)
    imSize = im0.size()
    spatial_filter, temp_size = _create_filters(filter_s_hsz, filter_t_hsz, opts, device=device)
    print '1. global normalization'
    if global_stat is None:
        if opts[0] == 0:
            # stat of the first image
            global_stat = [im0.mean(), im0.std()]
        else:
            raise Exception('need to implement')
    print '2. local normalization'
    mean_tensor = torch.zeros((imSize[0], imSize[1], temp_size), device=device)
    # initial chunk
    # e.g. sizeT=7, filterT_hsz=3, mid_r=3
    for i in range(filter_t_hsz + 1):  # 0-3
        # flip the initial frames to pad
        mean_tensor[:, :, i] = conv2d(_pre_process_ims(get_im(filter_t_hsz - i), global_stat, opts[0]),
                                      spatial_filter, padding=filter_s_hsz)
    for i in range(filter_t_hsz - 1):  # 0-1 -> 4-5
        mean_tensor[:, :, filter_t_hsz + 1 + i] = mean_tensor[:, :, filter_t_hsz - 1 - i].copy()
        # online change chunk
        print '2. local normalization'
        im_id = filter_t_hsz  # image
        chunk_id = temp_size - 1
        for i in range(num_slice):
            print 'process: %s/%s' % (i + 1, num_slice)
            # current frame
            im = _pre_process_ims(get_im(i), global_stat, opts[0])

            # last frame needed for temporal filter
            if filter_t_hsz + i < num_slice:
                imM = _pre_process_ims(get_im(filter_t_hsz + i), global_stat, opts[0])
            else:  # reflection mean
                imM = _pre_process_ims(get_im(num_slice - 1 - filter_t_hsz + (num_slice - 1 - i)), global_stat, opts[0])
                mean_tensor[:,:, chunk_id] = conv2d(imM, spatial_filter, padding=filter_s_hsz)

            # local temporal filtering
            if opts[2] == 0:  # median filter
                filterR = _3d_median_filter(mean_tensor, (1, 1, temp_size))
            else:
                raise Exception('need to implement')

            filterRD = filterR[:, :, filter_t_hsz] - mean_tensor[:, :, im_id]
            imDiff = conv2d(filterRD, spatial_filter, padding=filter_s_hsz)
            out_im = torch.clamp(im + imDiff, 0, 255).cpu().numpy()
            cv2.imwrite("output_%s" % (i), out_im)
            im_id = (im_id + 1) % temp_size
            chunk_id = (chunk_id + 1) % temp_size



