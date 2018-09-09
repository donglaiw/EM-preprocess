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
OUTPUT_FILE_DIR = "./test_output/"
DATASET_NAME = 'main'
SLICES = np.zeros((1, 1, 1))
FILTER_DIMS = [1, 1, 7]


def read_h5(directory, dataset_name, dtype=np.float32):
    return np.array(h5py.File(directory, 'r')[dataset_name], dtype=dtype)


def median_cpu(ims):
    return nd.median_filter(ims, FILTER_DIMS)


def median_gpu(ims):
    ims_cuda = torch.from_numpy(ims).cuda()
    filter_torch = torch.tensor(FILTER_DIMS, device='cpu', dtype=torch.float32) / 2
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
        SLICES = read_h5(INPUT_FILE_DIR, DATASET_NAME)
        if not os.path.exists(OUTPUT_FILE_DIR):
            os.makedirs(OUTPUT_FILE_DIR)
        dump_ims(OUTPUT_FILE_DIR + "original_%d.png", SLICES)

    def test_check_CUDA(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA-enabled GPU is not available.")

    def test_median_v1(self):
        profiler_cpu = profiler_gpu = cProfile.Profile()
        profiler_cpu.enable()
        out_cpu = median_cpu(SLICES)
        profiler_cpu.disable()
        profiler_gpu.enable()
        out_cuda = median_gpu(SLICES)
        profiler_gpu.disable()
        color_out = np.zeros((out_cpu.shape[0], out_cpu.shape[1], out_cpu.shape[2], 3))
        diff = out_cuda - out_cpu
        color_out[:, :, 0] = diff
        color_out[:, :, 1] = -diff
        dump_ims(OUTPUT_FILE_DIR + "cpu_v1_%d.png", out_cpu)
        dump_ims(OUTPUT_FILE_DIR + "gpu_v1_%d.png", out_cuda)
        dump_ims(OUTPUT_FILE_DIR + "diff_v1_%d.png", color_out)
        profiler_cpu.dump_stats(OUTPUT_FILE_DIR + "cpu_v1.profile")
        profiler_gpu.dump_stats(OUTPUT_FILE_DIR + "gpu_V1.profile")

    def test_minimal_median(self):
        profiler_cpu = profiler_gpu = cProfile.Profile()
        profiler_cpu.enable()
        out_cpu = out_cuda = []
        for i in range(len(SLICES) - FILTER_DIMS[2] / 2):
            profiler_cpu.enable()
            out_cpu.append(median_cpu(SLICES[i: i + FILTER_DIMS[2]]))
            profiler_cpu.disable()
            profiler_gpu.enable()
            out_cuda.append(median_gpu_v2(SLICES[i: i + FILTER_DIMS[2]]))
            profiler_gpu.disable()
        print out_cpu
        out_cpu = np.array(out_cpu)
        print out_cpu.shape
        out_cuda = np.array(out_cuda)
        color_out = np.zeros((out_cpu.shape[0], out_cpu.shape[1], 3))
        diff = out_cuda - out_cpu
        color_out[:, :, 0] = diff
        color_out[:, :, 1] = -diff
        dump_ims(OUTPUT_FILE_DIR + "cpu_v2_%d.png", out_cpu)
        dump_ims(OUTPUT_FILE_DIR + "gpu_v2_%d.png", out_cuda)
        dump_ims(OUTPUT_FILE_DIR + "diff_v2_%d.png", color_out)
        profiler_cpu.dump_stats(OUTPUT_FILE_DIR + "cpu_v2.profile")
        profiler_gpu.dump_stats(OUTPUT_FILE_DIR + "gpu_V2.profile")


if __name__ == '__main__':
    unittest.main()
