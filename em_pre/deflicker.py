import cv2
import numpy as np
import scipy.ndimage as nd
import pycuda.driver as cuda
import pycuda.autoinit
from multiprocessing import Queue
from multiprocessing import Pool
import os
import locked_dict

NUM_READ_ALLOC_PROCS = 32

def __np_arr_cuda_mem_alloc__(array):
    """
    Allocates the given numpy array on GPU memory.
    :param array: the desired numpy array.
    :return: the GPU memory reference when succeeds, -1 if a problem has been encountered.
    """
    try:
        shape = array.nbytes
        return cuda.mem_alloc(array.nbytes)
    except cuda.MemoryError:
        print 'Unable to allocate memory for numpy array.'
        return -1
    except AttributeError:
        print 'The passed argument is not a numpy array: %s.' % array
        return -1

def __pre_process__(img, global_stat=None, global_stat_opt=0):
    """
    :param img:
    :param global_stat:
    :param global_stat_opt:
    :return:
    """
    # im: x,y,t
    if global_stat_opt == 0:
        # mean/std
        if global_stat is not None:
            mm = img.mean(axis=0).mean(axis=0)
            if global_stat[1] > 0:
                tmp_std = np.std(img)
                if tmp_std < 1e-3:
                    img = (img - mm) + global_stat[0]
                else:
                    img = (img - mm) / np.std(img) * global_stat[1] + global_stat[0]
            else:
                if global_stat[0] > 0:
                    img = img - mm + global_stat[0]
    return img

def __load_pre_process_allocate__(get_img, idx, global_stat=None, global_stat_opt=0):
    ready_to_alloc_img = __pre_process__(img=get_img(idx), global_stat=global_stat, global_stat_opt=global_stat_opt)
    return __np_arr_cuda_mem_alloc__(ready_to_alloc_img)

def __create_filters__(filter_s_hsz, filter_t_hsz, opts):
    spat_size = [x * 2 + 1 for x in filter_s_hsz]
    temp_size = 2 * filter_t_hsz + 1
    if opts[1] == 0:  # mean filter: sum to 1
        spat_filter = np.ones(spat_size, dtype=np.float32) / (spat_size[0] * spat_size[1])
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
    ims = __pre_process__(ims.astype(np.float32), globalStat, opts[0])

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

def de_flicker_online(get_im, num_slice=100, opts=(0, 0, 0),
                      global_stat=None, filter_s_hsz=(15, 15), filter_t_hsz=2):
    # im: x,y,t
    # filters: spatial=(filterS_hsz, opts[1]), temporal=(filterT_hsz, opts[2])
    # seq stats: imSize, globalStat, globalStatOpt
    im0 = get_im(0)
    imSize = im0.shape
    spatial_filter, temp_size = __create_filters__(filter_s_hsz, filter_t_hsz, opts)
    #Create an empty output. Why?
    print '1. global normalization'
    if global_stat is None:
        if opts[0] == 0:
            # stat of the first image
            global_stat = [np.mean(im0), np.std(im0)]
        else:
            raise Exception('need to implement')
    print '2. local normalization'
    mean_tensor = np.zeros((imSize[0], imSize[1], temp_size))
    # initial chunk
    # e.g. sizeT=7, filterT_hsz=3, mid_r=3
    for i in range(filter_t_hsz + 1):  # 0-3
        # flip the initial frames to pad
        mean_tensor[:, :, i] = cv2.filter2D(__pre_process__(get_im(filter_t_hsz - i), global_stat, opts[0]),
                                           -1, spatial_filter, borderType=cv2.BORDER_REFLECT_101)
    for i in range(filter_t_hsz - 1):  # 0-1 -> 4-5
        mean_tensor[:, :, filter_t_hsz + 1 + i] = mean_tensor[:, :, filter_t_hsz - 1 - i].copy()


    # Step 1: Loading and pre-processing the image.
    try:
        reader_pid = os.fork()
    except OSError:
        print "Failed to create a child image producer process."
        return
    if reader_pid == 0: #The child process is in charge of reading and processing the images and feeding them into the gpu.
        single_worker_task = lambda x: __load_pre_process_allocate__(get_im, x, global_stat, opts[0])
        read_alloc_procs = Pool(NUM_READ_ALLOC_PROCS)
        read_alloc_procs.map(single_worker_task, range(num_slice))
        os._exit(0)
    else: #parent process
        # online change chunk
        print '2. local normalization'
        im_id = filter_t_hsz  # image
        chunk_id = temp_size - 1
        for i in range(num_slice):
            print 'process: %s/%s' % (i + 1, num_slice)
            # current frame
            im = __pre_process__(get_im(i), global_stat, opts[0])

            # last frame needed for temporal filter
            if filter_t_hsz + i < num_slice:
                imM = __pre_process__(get_im(filter_t_hsz + i), global_stat, opts[0])
            else:  # reflection mean
                imM = __pre_process__(get_im(num_slice - 1 - filter_t_hsz + (num_slice - 1 - i)), global_stat, opts[0])
            mean_tensor[:, :, chunk_id] = cv2.filter2D(imM, -1, spatial_filter, borderType=cv2.BORDER_REFLECT_101)

            # local temporal filtering
            if opts[2] == 0:  # median filter
                filterR = nd.filters.median_filter(mean_tensor, (1, 1, temp_size))
            else:
                raise Exception('need to implement')

            filterRD = filterR[:, :, filter_t_hsz] - mean_tensor[:, :, im_id]
            imDiff = cv2.filter2D(filterRD, -1, spatial_filter, borderType=cv2.BORDER_REFLECT_101)
            out_im = np.clip(im + imDiff, 0, 255).astype(np.uint8)
            cv2.imwrite("output_%s" % (i), out_im)
            im_id = (im_id + 1) % temp_size
            chunk_id = (chunk_id + 1) % temp_size



