"""
Name: slice_getters.py
Author: Matin Raayai Ardakani
An object that helps with loading, preprocessing and caching a stack of EM slices.
"""

from torch.multiprocessing import Pool, Lock, Manager, Condition
import torch
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
import h5py


class SliceGetter(object):


    def __init__(self, pre_processor, loader, idx_begin=0, idx_end=100, caching_limit=100):
        self.pre_processor = pre_processor
        self.loader = loader
        self.idx_begin = idx_begin
        self.idx_end = idx_end
        self.caching_limit = caching_limit
        manager = Manager()
        self.cache = manager.dict()
        self.cache_lock = self.idx_history_lock = Lock()
        self.cond_fill = Condition(self.cache_lock)
        self.cond_empty = Condition(self.cache_lock)
        self.idx_history = manager.Queue()
        self.stop_cleaner = False

    def start(self, num_processes):
        self.reader_procs = Pool(num_processes)
        self.reader_procs.map_async(self._reader_process, range(self.idx_begin, self.idx_end))

    def stop(self):
        self.stop_cleaner = True
        self.reader_procs.close()

    def _reader_process(self, idx):
        self.cache_lock.acquire()
        while len(self.cache == self.caching_limit):
            self.cond_empty.wait()
        if (not self.cache.has_key(idx)):
            self.cache[idx] = self.pre_processor(self.loader(idx))
        self.cond_fill.notify()
        self.cache_lock.release()

    def _cleaner_process(self):
        while not self.stop_cleaner:
            self.cache_lock.acquire()
            while len(self.cache) < self.caching_limit:
                self.cond_fill.wait()
            self.idx_history_lock.acquire()
            oldest_idx = self.idx_history.get()
            self.idx_history_lock.release()
            del self.cache[oldest_idx]
            self.cond_empty.notify()
            self.cache_lock.release()

    def __getitem__(self, item):
        """
        Gets the desired slice with the given index as a Pytorch tensor on the initialized device and datatype
        :param item: index of the desired item.
        :return: The desired slice as a Pytorch Tensor.
        """
        if item < self.idx_begin:
            return self.__getitem__(-item)
        if item > self.idx_end:
            return self.__getitem__(2 * self.idx_end - item)
        else:
            self.cache_lock.acquire()
            self.idx_history_lock.acquire()
            self.idx_history.put(item)
            self.idx_history_lock.release()
            if item in self.cache.keys():
                new_slice = self.cache[item]
            else:
                new_slice = self.pre_processor(self.loader(item))
                self.cache[item] = new_slice
            self.cache_lock.release()
            return new_slice


class DiskSliceGetter(SliceGetter):
    def __init__(self, path, pre_processor, device=None, idx_begin=0, idx_end=100, dtype=torch.float32, caching_limit=20):
        def loader(idx):
            slice_as_np = cv2.imread(path % idx).astype(self.dtype)
            return torch.from_numpy(slice_as_np).to(self.device)

        super(DiskSliceGetter, self).__init__(pre_processor, loader, device, idx_begin, idx_end, dtype, caching_limit)


class H5SliceGetter(SliceGetter):
    def __init__(self, path, pre_processor, data_set='main', device=None, idx_begin=0, idx_end=100, dtype=torch.float32,
                 caching_limit=20):
        def loader(idx):
            slice_as_np = np.array(h5py.File(path)[data_set][idx])
            return torch.from_numpy(slice_as_np).to(device)

        super(H5SliceGetter, self).__init__(pre_processor, loader, idx_begin, idx_end, caching_limit)




class CerebellumSliceGetter(SliceGetter):
    def __init__(self, path, pre_processor, device=None, idx_begin=0, idx_end=100, dtype=torch.float32, caching_limit=20):
        self.path = path
        super(CerebellumSliceGetter, self).__init__(pre_processor, device, idx_begin, idx_end, dtype, caching_limit)

    def loader(self, idx):
        pass
