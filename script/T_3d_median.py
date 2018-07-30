import numpy as np
import unittest
import torch
import em_pre_cuda
from T_util import writeh5
import h5py
import scipy.ndimage as nd

#Constants:
INPUT_FILE_DIR = '/n/coxfs01/donglai/data/SNEMI3D/train-input.h5'
OUTPUT_FILE_DIR = 'test_results'
DATASET_NAME = 'main'
BATCH_STACK_SIZE = 7
FILTER_DIMS = [1, 1, BATCH_STACK_SIZE]

def read_input(directory, dataset_name):
  return np.array(h5py.File(INPUT_FILE_DIR)[dataset_name])

def test_with_sikit(ims, filter_shape):
    return nd.median_filter(ims, filter_shape)
def test_with_cuda(ims, filter_shape):
    ims_cuda = torch.from_numpy(ims, device = 'cuda')
    output = ims_cuda.zeros_like(ims_cuda)
    em_pre_cuda.median_filter(ims_cuda, filter_shape)
    return output.numpy()

class TestMedian(unittest.TestCase):
  def test_median(self):
    self.assertTrue(torch.cuda.is_available())
    input_im_stack = read_input(INPUT_FILE_DIR, DATASET_NAME)
    out_cuda = np.zeros(input_im_stack.shape[0], input_im_stack.shape[1], input_im_stack.shape[2] / BATCH_STACK_SIZE)
    out_cpu = np.zeros(input_im_stack)
    for i in range(input_im_stack.shape[2] / BATCH_STACK_SIZE):
      cuda_result = test_with_sikit(input_im_stack, FILTER_DIMS)
      cpu_result = test_with_sikit(input_im_stack, FILTER_DIMS)
      out_cuda[:, :, i] = cuda_result.numpy()[:, :, BATCH_STACK_SIZE / 2]
      out_cpu[:, :, i] = cpu_result[:, :, BATCH_STACK_SIZE / 2]
    writeh5(OUTPUT_FILE_DIR + "cpu", DATASET_NAME, out_cpu)
    writeh5(OUTPUT_FILE_DIR + "gpu", DATASET_NAME, out_cuda)
    self.assertTrue(np.allclose(out_cpu, out_cuda))


if __name__ == '__main__':
  unittest.main()
