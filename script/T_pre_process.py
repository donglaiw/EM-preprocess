from em_pre_cuda.deflicker import _pre_process as p_cuda
from em_pre.deflicker import preprocess as p_cpu
import unittest
import torch
import numpy as np
import cv2
from T_util import read_h5
# Constants:
INPUT_FILE_DIR = '/home/matinraayai/Data/test_data.h5'
DATASET_NAME = 'main'





class TestPreprocess(unittest.TestCase):

    def test_check_cuda(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA-enabled GPU is not available.")

    def test_median(self):
        in_ims = read_h5(INPUT_FILE_DIR, DATASET_NAME)
        diff = []
        for i, img in enumerate(in_ims):
            out_cpu = torch.from_numpy(p_cpu(img, globalStat=(150, -1), globalStatOpt=0))
            out_gpu = p_cuda(torch.from_numpy(img).cuda(), global_stat=(150, -1), method='naive').cpu()
            cv2.imwrite("preprocess%d.png" % (i + 1), (out_cpu - out_gpu).numpy())
            diff.append((out_gpu - out_cpu).numpy())
        self.assertTrue(np.array(diff).sum() == 0 or (np.array(diff).sum() < 50 and -50 < np.array(diff).sum()))



if __name__ == '__main__':
    unittest.main()
