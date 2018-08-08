"""*********************************************************************************************************************
 * Name: T_3d_median.py
 * Author: Matin Raayai Ardakani
 * Email: raayai.matin@gmail.com
 * Unit test and profiler for em_pre_cuda's median_filter function.
 ********************************************************************************************************************"""
import numpy as np
import unittest
import torch
import em_pre_cuda
from T_util import writeh5
import h5py
import scipy.ndimage as nd
import cProfile
import cv2

profiler_cpu = cProfile.Profile()
profiler_gpu = cProfile.Profile()

#Constants:
INPUT_FILE_DIR = 'test_data.h5'
OUTPUT_FILE_DIR = 'Test Results'
DATASET_NAME = 'main'
BATCH_STACK_SIZE = 5
FILTER_DIMS = 5

def read_input(directory, dataset_name):
    return np.array(h5py.File(directory, 'r')[dataset_name], dtype=np.float32)
    # out = np.zeros((100, 1000, 1000), dtype = np.float)
    # for i in range(99):
    #     out[i, :, :] = np.ones((1000, 1000), dtype = np.float) * i * 2
    # return out

def test_with_sikit(ims, filter_shape):
    output = nd.median_filter(ims, filter_shape)
    print output
    return output
def test_with_cuda(ims, filter_shape):
    ims_cuda = torch.from_numpy(ims).cuda()
    print ims_cuda.size()
    filter_torch = torch.tensor(filter_shape / 2, device = 'cuda', dtype=torch.float32)
    output = em_pre_cuda.median_filter(ims_cuda, torch.tensor([10., 1., 1.], device='cpu', dtype=torch.float32))
    #print output
    print "Finished Cuda."
    return output.cpu().numpy()

class TestMedian(unittest.TestCase):
  def test_median(self):
    self.assertTrue(torch.cuda.is_available(), "CUDA-enabled GPU is not available.")
    input_im_stack = read_input(INPUT_FILE_DIR, DATASET_NAME)
    #input_im_stack = input_im_stack.transpose(1, 2, 0)
    output_im_shape = (input_im_stack.shape[0], input_im_stack.shape[1], input_im_stack.shape[2])
    out_cuda = np.zeros(output_im_shape)
    out_cpu = np.zeros(output_im_shape)
    profiler_cpu.enable()
    out_cpu = test_with_sikit(input_im_stack, [10, 1, 1])
    print out_cpu.shape
    profiler_cpu.disable()
    profiler_gpu.enable()
    out_cuda = test_with_cuda(input_im_stack, FILTER_DIMS)
    print out_cuda.shape
    profiler_gpu.disable()
    for i in range(len(out_cpu)):
        cv2.imwrite("cpu_%d.png" % (i + 1), out_cpu[i])
        cv2.imwrite("gpu_%d.png" % (i + 1), out_cuda[i])

    #for i in range(input_im_stack.shape[0] / BATCH_STACK_SIZE):
    #  print "Image %d/%d" % (i + 1, input_im_stack.shape[0] / BATCH_STACK_SIZE)
    #  profiler_gpu.enable()
    #  cuda_result = test_with_cuda(input_im_stack[i: i + BATCH_STACK_SIZE], FILTER_DIMS)
    #  profiler_gpu.disable()
    #  profiler_cpu.enable()
    #  cpu_result = test_with_sikit(input_im_stack[i: i + BATCH_STACK_SIZE], FILTER_DIMS)
    #  profiler_cpu.disable()
    #  out_cuda[i, :, :] = cuda_result[BATCH_STACK_SIZE / 2, :, :]
    #  out_cpu[i, :, :] = cpu_result[BATCH_STACK_SIZE / 2, :, :]
    #writeh5(OUTPUT_FILE_DIR + "_cpu.h5", DATASET_NAME, out_cpu)
    #writeh5(OUTPUT_FILE_DIR + "_gpu.h5", DATASET_NAME, out_cuda)
    profiler_cpu.dump_stats("cpu.profile")
    profiler_gpu.dump_stats("gpu.profile")
    #self.assertTrue(np.allclose(out_cpu, out_cuda))


if __name__ == '__main__':
  unittest.main()