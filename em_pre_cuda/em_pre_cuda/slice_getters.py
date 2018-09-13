"""
Name: slice_getters.py
Author: Matin Raayai Ardakani
An object that helps with loading, preprocessing and caching a stack of EM slices.
"""

from threading import Lock, Thread, Condition
from multiprocessing.pool import ThreadPool
from torch.multiprocessing import Queue
import torch
import cv2
import numpy as np
import h5py


class SliceGetter(object):
    def __init__(self, pre_processor, loader, num_threads=20, idx_begin=0, idx_end=100, caching_limit=100):
        self.pre_processor = pre_processor
        self.loader = loader
        self.idx_begin = idx_begin
        self.idx_end = idx_end
        self.caching_limit = caching_limit
        self.cache = {}
        self.cond_empty = self.cond_fill = Condition()
        self.idx_history = Queue()
        self.reader_threads = ThreadPool(num_threads)
        self.stop_cleaner = False
        self.cleaner_threads = Thread(target=self.cleaner_thread)
        self.cleaner_threads.daemon = True

    def start(self):
        self.reader_threads.map_async(self.reader_thread, range(self.idx_begin, self.idx_end))
        self.cleaner_threads.start()

    def stop(self):
        self.stop_cleaner = True
        self.reader_threads.close()

    def reader_thread(self, idx):
        self.cond_empty.acquire()
        while len(self.cache) >= self.caching_limit:
            self.cond_empty.wait()
        if idx not in self.cache:
            self.cache[idx] = self.pre_processor(self.loader(idx))
        self.cond_fill.notify()
        self.cond_empty.release()

    def cleaner_thread(self):
        self.cond_fill.acquire()
        while len(self.cache) < self.caching_limit:
            self.cond_fill.wait()
        oldest_idx = self.idx_history.get()
        del self.cache[oldest_idx]
        self.cond_empty.notify()
        self.cond_fill.release()

    def __getitem__(self, item):
        """
        Gets the desired slice with the given index as a Pytorch tensor on the initialized device and datatype
        :param item: index of the desired item.
        :return: The desired slice as a Pytorch Tensor.
        """
        if item < self.idx_begin:
            return self.__getitem__(-item)
        if item > self.idx_end - 1:
            return self.__getitem__(2 * self.idx_end - item - 2)
        else:
            self.idx_history.put(item)
            if item in self.cache.keys():
                new_slice = self.cache[item]
            else:
                new_slice = self.pre_processor(self.loader(item))
                self.cache[item] = new_slice
            return new_slice


def h5_slice_getter(path, pre_processor, num_threads=5, data_set='main', device='cpu', idx_begin=0, idx_end=100,
                    dtype=np.float32, caching_limit=100):
    lk = Lock()

    def loader(idx):
        #lk.acquire()
        slice_as_np = np.array(h5py.File(path)[data_set][idx], dtype=dtype)
        #lk.release()
        return torch.from_numpy(slice_as_np).to(device)
    return SliceGetter(pre_processor, loader, num_threads, idx_begin, idx_end, caching_limit)


def png_slice_getter(path, pre_processor, num_threads, device='cpu', idx_begin=0, idx_end=100, dtype=np.float32,
                     caching_limit=20):
    def loader(idx):
        slice_as_np = cv2.imread(path % idx).astype(dtype)
        return torch.from_numpy(slice_as_np).to(device)
    return SliceGetter(pre_processor, loader, num_threads, idx_begin, idx_end, caching_limit)


def cerebellum_slice_getter(orig_path, pre_processor, num_threads, device='cpu', idx_begin=0, idx_end=100,
                            dtype=np.float32, caching_limit=20):
    def loader(idx):
        raise NotImplementedError()
    return SliceGetter(pre_processor, loader, num_threads, idx_begin, idx_end, caching_limit)
