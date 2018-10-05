"""*********************************************************************************************************************
 * Name: T_torch_extension.py
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Unit test and profiler for em_pre_cuda's median_filter function.
 ********************************************************************************************************************"""
import numpy as np
import unittest
import torch
import em_pre_torch_ext
from T_util import writeh5
import h5py
import scipy.ndimage as nd
import cProfile
import cv2

profiler_cpu = cProfile.Profile()
profiler_gpu = cProfile.Profile()

# Constants:
INPUT_FILE_DIR = '/home/matinraayai/cerebellum_test_chunk.h5'
OUTPUT_FILE_DIR = 'Test Results'
DATASET_NAME = 'main'
BATCH_STACK_SIZE = 5
FILTER_DIMS = [11, 1, 1]


class TestMedian(unittest.TestCase):

    def _read_h5(self, directory, dataset_name, dtype=np.float32):
        return np.array(h5py.File(directory, 'r')[dataset_name], dtype=dtype)

    def _median_cpu(self, ims, flt_shape):
        return nd.median_filter(ims, flt_shape)

    def _median_cuda(self, ims, flt_shape):
        ims_cuda = torch.from_numpy(ims).cuda()
        filter_torch = torch.tensor(flt_shape) / 2
        output = em_pre_torch_ext.median_filter(ims_cuda, filter_torch)
        return output.cpu().numpy()

    def test_check_cuda(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA-enabled GPU is not available.")

    def test_median(self):
        in_ims = self._read_h5(INPUT_FILE_DIR, DATASET_NAME)
        out_ims_shape = in_ims.shape
        profiler_cpu.enable()
        out_cpu = self._median_cpu(in_ims, FILTER_DIMS)
        profiler_cpu.disable()
        profiler_gpu.enable()
        out_cuda = self._median_cuda(in_ims, FILTER_DIMS)
        profiler_gpu.disable()
        for i in range(len(out_cpu)):
            cv2.imwrite("original_%d.png" % (i + 1), in_ims[i])
            cv2.imwrite("cpu_%d.png" % (i + 1), out_cpu[i])
            cv2.imwrite("gpu_%d.png" % (i + 1), out_cuda[i])
        profiler_cpu.dump_stats("cpu.profile")
        profiler_gpu.dump_stats("gpu.profile")


if __name__ == '__main__':
    unittest.main()
