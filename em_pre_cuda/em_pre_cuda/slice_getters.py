"""
Name: slice_getters.py
Author: Matin Raayai Ardakani
An object that helps with loading, preprocessing and caching a stack of EM slices.
TODO: Make threadsafe.
"""
import torch
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
import h5py


class SliceGetter:
    __metaclass__ = ABCMeta

    def __init__(self, path, pre_processor, device=None, idx_begin=0, idx_end=100, dtype=np.float32, caching_limit=20):
        self.path = path
        self.pre_processor = pre_processor
        self.device = torch.device(device) if device is not None else 'cpu'
        self.idx_begin = idx_begin
        self.idx_end = idx_end
        self.dtype = dtype
        self.caching_limit = caching_limit
        self.last_accessed_idx = idx_begin
        self.cache = {}

    @abstractmethod
    def __load_from_disk(self, idx):
        pass

    def __getitem__(self, item):
        """
        Gets the desired slice with the given index as a Pytorch tensor on the initialized device and datatype
        :param idx:
        :return:
        """
        if item < self.idx_begin:
            return self.__getitem__(-item)
        if item > self.idx_end:
            return self.__getitem__(2 * self.idx_end - item)
        else:
            self.last_accessed_idx = item
            if item in self.cache.keys():
                return self.cache[item]
            else:
                new_slice = self.pre_processor(self.__load_from_disk(item))
                self.cache[item] = new_slice
                return new_slice

    def remove_from_cache(self, idx):
        del self.cache[idx]


class DiskSliceGetter(SliceGetter):
    def __init__(self, path, pre_processor, device=None, idx_begin=0, idx_end=100, dtype=np.float32, caching_limit=20):
        super(DiskSliceGetter, self).__init__(path, pre_processor, device, idx_begin, idx_end, dtype, caching_limit)

    def __load_from_disk(self, idx):
        slice_as_np = cv2.imread(self.path % idx).astype(self.dtype)
        return torch.from_numpy(slice_as_np).to(self.device)


class H5SliceGetter(SliceGetter):
    def __init__(self, path, pre_processor, data_set='main', device=None, idx_begin=0, idx_end=100, dtype=np.float32,
                 caching_limit=20):
        self.data_set = data_set
        super(H5SliceGetter, self).__init__(path, pre_processor, device, idx_begin, idx_end, dtype, caching_limit)

    def __load_from_disk(self, idx):
        slice_as_np = np.array(h5py.File(self.path)[self.data_set][idx])
        return torch.from_numpy(slice_as_np).to(self.device)


class CerebellumSliceGetter(SliceGetter):
    def __init__(self, path, pre_processor, device=None, idx_begin=0, idx_end=100, dtype=np.float32, caching_limit=20):
        super(CerebellumSliceGetter, self).__init__(path, pre_processor, device, idx_begin, idx_end, dtype,
                                                    caching_limit)

    def __load_from_disk(self, idx):
        pass
