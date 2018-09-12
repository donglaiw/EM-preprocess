from em_pre_cuda.slice_getters import H5SliceGetter
from em_pre_cuda.pre_process import NaivePreProcess
import h5py
import torch
import cv2

INPUT_FILE_PATH = "/home/matinraayai/cerebellum_test_chunk.h5"

s_getter = H5SliceGetter(INPUT_FILE_PATH, NaivePreProcess((150, -1)), 'main', 'cpu')
s_getter.start(5)

for i in range(100):
    print i
    cv2.imwrite("getter_%d.png" % i, s_getter[i])