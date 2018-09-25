"""*********************************************************************************************************************
 * Name: T_torch_extension.py
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Unit test and profiler for em_pre_cuda's median_filter function.
 ********************************************************************************************************************"""
import os

import numpy as np
import unittest
import torch
import em_pre_torch_ext
from T_util import writeh5
import h5py
import scipy.ndimage as nd
import cProfile
import cv2
import sys

# Constants:
INPUT_FILE_DIR = "/home/matinraayai/cerebellum_test_chunk.h5"
OUTPUT_FILE_DIR = "test_output/"
DATASET_NAME = 'main'
SLICES = np.zeros((100, 1024, 1024))
FILTER_DIMS = [7, 1, 1]


def read_h5(directory, dataset_name, dtype=np.float32):
    return np.array(h5py.File(directory, 'r')[dataset_name], dtype=dtype)


def median_cpu(ims):
    return nd.median_filter(ims, FILTER_DIMS)


def median_gpu(ims):
    ims_cuda = torch.from_numpy(ims).cuda()
    filter_torch = [i / 2 for i in FILTER_DIMS]
    output = em_pre_torch_ext.median_filter(ims_cuda, filter_torch)
    return output.cpu().numpy()


def median_gpu_v2(ims):
    g_ims = torch.from_numpy(ims).cuda()
    return em_pre_torch_ext.median_filter(g_ims).cpu().numpy()


def dump_ims(path, ims):
    for i in range(len(ims)):
        cv2.imwrite(path % (i + 1), ims[i])


class TestMedian(unittest.TestCase):

    def test_read_h5(self):
        global SLICES
        SLICES = read_h5(INPUT_FILE_DIR, DATASET_NAME)
        if not os.path.exists(OUTPUT_FILE_DIR):
            os.makedirs(OUTPUT_FILE_DIR)
        dump_ims(OUTPUT_FILE_DIR + "original_%d.png", SLICES)

    def test_check_CUDA(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA-enabled GPU is not available.")

    def test_median_v1(self):
        slices = read_h5(INPUT_FILE_DIR, DATASET_NAME)
        profiler_cpu = cProfile.Profile()
        profiler_gpu = cProfile.Profile()
        profiler_cpu.enable()
        out_cpu = median_cpu(slices)
        profiler_cpu.disable()
        profiler_gpu.enable()
        out_cuda = median_gpu(slices)
        profiler_gpu.disable()
        color_out = np.zeros((out_cpu.shape[0], out_cpu.shape[1], out_cpu.shape[2], 3))
        diff = out_cuda - out_cpu
        color_out[:, :, :, 0] = diff
        color_out[:, :, :, 1] = -diff
        dump_ims(OUTPUT_FILE_DIR + "cpu_v1_%d.png", out_cpu)
        dump_ims(OUTPUT_FILE_DIR + "gpu_v1_%d.png", out_cuda)
        dump_ims(OUTPUT_FILE_DIR + "diff_v1_%d.png", color_out)
        profiler_cpu.print_stats(1)
        print "Test 1:"
        profiler_gpu.print_stats(1)
        profiler_cpu.dump_stats(OUTPUT_FILE_DIR + "cpu_v1.profile")
        profiler_gpu.dump_stats(OUTPUT_FILE_DIR + "gpu_V1.profile")

    def test_minimal_median(self):
        slices = read_h5(INPUT_FILE_DIR, DATASET_NAME)
        profiler_cpu = cProfile.Profile()
        profiler_gpu = cProfile.Profile()
        profiler_cpu.enable()
        out_cpu = out_cuda = []
        for i in range(len(slices) - FILTER_DIMS[0] / 2 - 3):
            profiler_cpu.enable()
            out_cpu.append(median_cpu(slices[i: i + FILTER_DIMS[0]])[FILTER_DIMS[0] / 2])
            profiler_cpu.disable()
            profiler_gpu.enable()
            out_cuda.append(median_gpu_v2(slices[i: i + FILTER_DIMS[0]]))
            profiler_gpu.disable()
        out_cpu = np.array(out_cpu)
        out_cuda = np.array(out_cuda)
        color_out = np.zeros((out_cpu.shape[0], out_cpu.shape[1], out_cpu.shape[2], 3))
        diff = out_cuda - out_cpu
        color_out[:, :, :, 0] = diff
        color_out[:, :, :, 1] = -diff
        dump_ims(OUTPUT_FILE_DIR + "cpu_v2_%d.png", out_cpu)
        dump_ims(OUTPUT_FILE_DIR + "gpu_v2_%d.png", out_cuda)
        dump_ims(OUTPUT_FILE_DIR + "diff_v2_%d.png", color_out)
        print "Test 2:"
        profiler_cpu.print_stats(1)
        profiler_gpu.print_stats(1)
        profiler_cpu.dump_stats(OUTPUT_FILE_DIR + "cpu_v2.profile")
        profiler_gpu.dump_stats(OUTPUT_FILE_DIR + "gpu_V2.profile")


if __name__ == '__main__':
    unittest.main()
