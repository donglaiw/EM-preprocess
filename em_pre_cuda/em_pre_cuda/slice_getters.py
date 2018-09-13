"""
Name: slice_getters.py
Author: Matin Raayai Ardakani
An object that helps with loading, preprocessing and caching a stack of EM slices.
"""

from torch.multiprocessing import Pool, Manager, Queue, Lock
import torch
import cv2
import numpy as np
import h5py


class SliceGetter(object):
    def __init__(self, pre_processor, loader, num_processes=5, idx_begin=0, idx_end=100, caching_limit=100):
        self.pre_processor = pre_processor
        self.loader = loader

        self.idx_begin = idx_begin
        self.idx_end = idx_end
        self.caching_limit = caching_limit
        manager = Manager()
        self.cache = manager.dict()
        self.cond_empty = self.cond_fill = manager.Condition()
        self.idx_history = Queue()
        self.reader_procs = Pool(num_processes)
        self.stop_cleaner = False

    def start(self):
        self.reader_procs.map_async(self._reader_process, range(self.idx_begin, self.idx_end))

    def stop(self):
        self.stop_cleaner = True
        self.reader_procs.close()

    def _reader_process(self, idx):
        self.cond_empty.acquire()
        while len(self.cache == self.caching_limit):
            self.cond_empty.wait()
        if idx not in self.cache:
            self.cache[idx] = self.pre_processor(self.loader(idx))
        self.cond_fill.notify()
        self.cond_empty.release()

    def _cleaner_process(self):
        self.cond_fill.acquire()
        while not self.stop_cleaner:
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


def h5_slice_getter(path, pre_processor, num_processes=5, data_set='main', device='cpu', idx_begin=0, idx_end=100,
                    dtype=np.float32, caching_limit=100):
    lk = Lock()

    def loader(idx):
        lk.acquire()
        slice_as_np = np.array(h5py.File(path)[data_set][idx], dtype=dtype)
        lk.release()
        return torch.from_numpy(slice_as_np).to(device)
    return SliceGetter(pre_processor, loader, num_processes, idx_begin, idx_end, caching_limit)


def png_slice_getter(path, pre_processor, num_processes, device='cpu', idx_begin=0, idx_end=100, dtype=np.float32,
                     caching_limit=20):
    def loader(idx):
        slice_as_np = cv2.imread(path % idx).astype(dtype)
        return torch.from_numpy(slice_as_np).to(device)
    return SliceGetter(pre_processor, loader, num_processes, idx_begin, idx_end, caching_limit)


def cerebellum_slice_getter(orig_path, pre_processor, num_processes, device='cpu', idx_begin=0, idx_end=100,
                            dtype=np.float32, caching_limit=20):
    def loader(idx):
        raise NotImplementedError()
    return SliceGetter(pre_processor, loader, num_processes, idx_begin, idx_end, caching_limit)
