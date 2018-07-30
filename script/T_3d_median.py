import numpy as np
import unittest
import torch
import em_pre_cuda
from T_util import writeh5
import h5py
import scipy.ndimage as nd
import cProfile

profiler_cpu = cProfile.Profile()
profiler_gpu = cProfile.Profile()



#Constants:
INPUT_FILE_DIR = 'data.h5'
OUTPUT_FILE_DIR = 'test_results'
DATASET_NAME = 'main'
BATCH_STACK_SIZE = 7
FILTER_DIMS = [1, 1, BATCH_STACK_SIZE]

def read_input(directory, dataset_name):
  return np.array(h5py.File(INPUT_FILE_DIR)[dataset_name], dtype=np.float32)

def test_with_sikit(ims, filter_shape):
    return nd.median_filter(ims, filter_shape)
def test_with_cuda(ims, filter_shape):
    ims_cuda = torch.from_numpy(ims).cuda()
    filter_torch = torch.tensor(filter_shape, device = 'cuda', dtype=torch.float32)
    output = em_pre_cuda.median_filter(ims_cuda, filter_torch)
    return output.cpu().numpy()

class TestMedian(unittest.TestCase):
  def test_median(self):
    self.assertTrue(torch.cuda.is_available())
    input_im_stack = read_input(INPUT_FILE_DIR, DATASET_NAME)
    out_cuda = np.zeros((input_im_stack.shape[0] / BATCH_STACK_SIZE, input_im_stack.shape[1], input_im_stack.shape[2]))
    print out_cuda.shape
    out_cpu = np.zeros((input_im_stack.shape[0] / BATCH_STACK_SIZE, input_im_stack.shape[1], input_im_stack.shape[2]))
    for i in range(input_im_stack.shape[0] / BATCH_STACK_SIZE):
      print "Image %d/%d" % (i + 1, input_im_stack.shape[0] / BATCH_STACK_SIZE)
      profiler_gpu.enable()
      cuda_result = test_with_cuda(input_im_stack[i: i + BATCH_STACK_SIZE], FILTER_DIMS)
      profiler_gpu.disable()
      profiler_cpu.enable()
      cpu_result = test_with_sikit(input_im_stack[i: i + BATCH_STACK_SIZE], FILTER_DIMS)
      profiler_cpu.disable()
      out_cuda[i, :, :] = cuda_result[BATCH_STACK_SIZE / 2, :, :]
      out_cpu[i, :, :] = cpu_result[BATCH_STACK_SIZE / 2, :, :]
    writeh5(OUTPUT_FILE_DIR + "_cpu.h5", DATASET_NAME, out_cpu)
    writeh5(OUTPUT_FILE_DIR + "_gpu.h5", DATASET_NAME, out_cuda)
    profiler_cpu.dump_stats("cpu.profile")
    profiler_gpu.dump_stats("gpu.profile")
    self.assertTrue(np.allclose(out_cpu, out_cuda))


if __name__ == '__main__':
  unittest.main()
