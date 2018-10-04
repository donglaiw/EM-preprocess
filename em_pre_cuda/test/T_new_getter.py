from em_pre_cuda.slice_getters import h5_slice_getter
from em_pre_cuda.pre_process import ThresholdPreProcess
from em_pre_cuda.spatial import PyTorch2dConvolution
from em_pre_cuda.temporal import PyTorchExtMedian
from em_pre_cuda import deflicker
from em_pre_cuda.loader import Manager
from torch.multiprocessing import Pool
import h5py
import torch
import cv2
import cProfile
import numpy as np

profiler = cProfile.Profile()
def loader(idx):
    print idx
    return torch.tensor(h5py.File('/home/matinraayai/cerebellum_test_chunk.h5')['main'][idx], dtype=torch.float32)

def saver(idx, slc):
    cv2.imwrite("result_%d.png" % idx, slc)

manager = Manager(loader, saver, range(100))
manager.start()
manager.wait()
profiler.disable()
profiler.print_stats(1)
"""
s_getter.stop()
"""