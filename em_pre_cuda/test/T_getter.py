from em_pre_cuda.slice_getters import h5_slice_getter
from em_pre_cuda.pre_process import ThresholdPreProcess
from em_pre_cuda.spatial import PyTorch2dConvolution
from em_pre_cuda.temporal import PyTorchExtMedian
from em_pre_cuda import deflicker
from em_pre_cuda.writer import png_slice_writer
from torch.multiprocessing import Pool
import h5py
import torch
import cv2
import cProfile

profiler = cProfile.Profile()

INPUT_FILE_PATH = "/home/matinraayai/cerebellum_test_chunk.h5"
s_getter = h5_slice_getter(INPUT_FILE_PATH, ThresholdPreProcess((150, -1)), 200, 'main', 'cuda:0', caching_limit=20)
s_filter = PyTorch2dConvolution(15, 'cuda:0')
t_filter = PyTorchExtMedian()
s_getter.start()
"""
for i in range(100):
    print i
    cv2.imwrite("getter_%d.png" % i, s_getter[i].cpu().numpy())
"""


s_writer = png_slice_writer('./', '%d.png', 10)
s_writer.start()


def target(idx):
    #print idx
    s_writer.post(deflicker(s_getter, s_filter, t_filter, range(idx - 2, idx + 3)).cpu(), idx)

"""
p = Pool(1)
result = p.map(target, range(100))
"""


profiler.enable()
for i in range(100):
    print i
    target(i - 1)
    # cv2.imwrite("result_%d.png" % i, target(i - 1).cpu().numpy())
profiler.disable()
profiler.print_stats(1)
"""
s_getter.stop()
"""