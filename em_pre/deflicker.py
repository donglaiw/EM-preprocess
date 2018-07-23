import cv2
import numpy as np
import scipy.ndimage as nd
import pycuda.driver as cuda
import pycuda.autoinit
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock
import os
NUM_BATCH_READ_PROCS = 4
NUM_READ_ALLOC_PROCS = 32
#TODO A sempahore to control the loading progress.
class ImgHToDLoader(Thread):
    def __init__(self, getter, start_idx, end_idx, originals, originals_lock, filtered_sets, filtered_lock,
                 global_stat = None,
                 global_stat_opt = 0):
        Thread.__init__(self)
        self.getter = getter
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.originals = originals
        self.originals_lock = originals_lock
        self.filtered_lock = filtered_lock
        self.filtered_sets = filtered_sets
        self.global_stat = global_stat
        self.global_stat_opt = global_stat_opt

    def _load_ims(self):
        self.ims = ThreadPool(NUM_BATCH_READ_PROCS).map(self.getter, range(self.start_idx, self.end_idx))
    def _pre_process_ims(self):
        # im: x,y,t
        if self.global_stat_opt == 0:
            # mean/std
            if self.global_stat is not None:
                mm = self.ims.mean(axis=0).mean(axis=0)
                if self.global_stat[1] > 0:
                    tmp_std = np.std(self.ims)
                    if tmp_std < 1e-3:
                        self.ims = (self.ims - mm) + self.global_stat[0]
                    else:
                        self.ims = (self.ims - mm) / np.std(self.ims) * self.global_stat[1] + self.global_stat[0]
                else:
                    if self.global_stat[0] > 0:
                        self.ims = self.ims - mm + self.global_stat[0]
        return self.ims

    def _single_img_htod(self, img):
        img_d = cuda.mem_alloc(img.nbytes)
        cuda.memcpy_htod(img_d, img)
        return img_d

    def _batch_img_htod(self):
        num_bytes = self.ims[0].nbytes
        return ThreadPool(NUM_BATCH_READ_PROCS).map(self._single_img_htod, self.ims)

    def _write_to_dict(self, dict, lock):
        lock.aquire()
        for i, img_d in enumerate(self.batch_d):
            dict[i + self.start_idx] = img_d
        lock.release()

    #TODO Import the CV2 kernel for this.
    def _filter2d_cuda_single(self, img_d):
        return



    def _filter2d_cuda_batch(self):
        return ThreadPool(NUM_BATCH_READ_PROCS).map(self._filter2d_cuda_single, self.batch_d)


    def run(self):
        self._load_ims()
        self._pre_process_ims()
        self.batch_d = self._batch_img_htod()
        self._write_to_dict(self.originals, self.originals_lock)
        self._filter2d_cuda_batch()
        self._write_to_dict(self.filtered_sets, self.filtered_lock)
        return

def _create_filters(filter_s_hsz, filter_t_hsz, opts):
    spat_size = [x * 2 + 1 for x in filter_s_hsz]
    temp_size = 2 * filter_t_hsz + 1
    if opts[1] == 0:  # mean filter: sum to 1
        spat_filter = np.ones(spat_size, dtype=np.float32) / (spat_size[0] * spat_size[1])
    else:
        raise Exception('need to implement')
    return spat_filter, temp_size

def _fork():
    try:
        return os.fork()
    except OSError:
        print "Failed to create a child image producer process."
        return

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
    ims = _pre_process(ims.astype(np.float32), globalStat, opts[0])

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
    spatial_filter, temp_size = _create_filters(filter_s_hsz, filter_t_hsz, opts)
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
        mean_tensor[:, :, i] = cv2.filter2D(_pre_process(get_im(filter_t_hsz - i), global_stat, opts[0]),
                                            -1, spatial_filter, borderType=cv2.BORDER_REFLECT_101)
    for i in range(filter_t_hsz - 1):  # 0-1 -> 4-5
        mean_tensor[:, :, filter_t_hsz + 1 + i] = mean_tensor[:, :, filter_t_hsz - 1 - i].copy()


    # Step 1: Loading and pre-processing the image.


    reader_pid = _fork()
    if reader_pid == 0: #The child process reads and processes the images and feeds them into the gpu.
        for i in range(num_slice / temp_size):
            current_batch = batch_img_loader(get_im, i * temp_size, temp_size)
            Process(target=_load_pre_process_allocate, args=(current_batch, global_stat, opts[0])).start()
        last_batch = batch_img_loader(get_im, temp_size * (num_slice / temp_size), num_slice % temp_size)
        Process(target=_load_pre_process_allocate, args=(last_batch, global_stat, opts[0])).start()
        os._exit(0)
    else: #parent process
        originals = {}
        filtered = {}
        orig_lock = Lock()
        filt_lock = Lock()
        htod_loader_threads = [ImgHToDLoader(getter=get_im,
                                             start_idx=i * temp_size,
                                             end_idx=(i + 1) * temp_size,
                                             originals=originals,
                                             originals_lock=orig_lock,
                                             filtered_sets=filtered,
                                             filtered_lock=filt_lock,
                                             global_stat=global_stat,
                                             global_stat_opt=opts[0]) for i in range(num_slice / temp_size) ]
        htod_loader_threads.append(ImgHToDLoader(getter=get_im,
                                                 start_idx=temp_size * (num_slice / temp_size),
                                                 end_idx=num_slice,
                                                 originals=originals,
                                                 originals_lock=orig_lock,
                                                 filtered_sets=filtered,
                                                 filtered_lock=filt_lock,
                                                 global_stat=global_stat,
                                                 global_stat_opt=opts[0]))
        # online change chunk
        print '2. local normalization'
        im_id = filter_t_hsz  # image
        chunk_id = temp_size - 1
        for i in range(num_slice):
            print 'process: %s/%s' % (i + 1, num_slice)
            # current frame
            im = _pre_process(get_im(i), global_stat, opts[0])

            # last frame needed for temporal filter
            if filter_t_hsz + i < num_slice:
                imM = _pre_process(get_im(filter_t_hsz + i), global_stat, opts[0])
            else:  # reflection mean
                imM = _pre_process(get_im(num_slice - 1 - filter_t_hsz + (num_slice - 1 - i)), global_stat, opts[0])
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



