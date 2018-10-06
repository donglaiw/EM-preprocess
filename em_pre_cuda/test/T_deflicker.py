import cProfile
import sys
from em_pre_cuda.pre_process import ThresholdPreProcess
from em_pre_cuda.spatial import PyTorch2dConvolution
from em_pre_cuda.temporal import PyTorchExtMedian, NdImageMedian
from em_pre_cuda import de_flicker
import h5py
import torch
import cv2

DEVICE = sys.argv[4] # Either 'cpu' or 'gpu'
INPUT_FILE_PATH = "/n/coxfs01/donglai/ppl/matin/data_h5/cerebellum_test_chunk.h5"
OUTPUT_FILE_PATH = "/n/coxfs01/donglai/ppl/matin/test_output/df_%s_%d.png"
PROFILER_OUTPUT_PATH = "/n/coxfs01/donglai/ppl/matin/test_output/df_%s_%d.png"
MEAN_FILTER_RAD = 15
MEDIAN_FILTER_RAD = 2
SLICE_RANGE = range(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

pp = ThresholdPreProcess((150, -1))
s_filter = PyTorch2dConvolution(MEAN_FILTER_RAD)
t_filter = PyTorchExtMedian() if DEVICE == 'cuda' else NdImageMedian()
profiler = cProfile.Profile()

fd = h5py.File(INPUT_FILE_PATH)['main']
def im_read(idx):
    return torch.tensor(fd[idx], torch.float32).to(DEVICE)

profiler.enable()
init_slc_range = SLICE_RANGE[0: MEDIAN_FILTER_RAD]
init_slc_range.reverse()
slc_window = [pp(im_read(i).to(DEVICE)) for i in init_slc_range]
for i in range(MEDIAN_FILTER_RAD - 1, 0, -1):
    slc_window.append(slc_window[i])
for i in SLICE_RANGE[:-MEDIAN_FILTER_RAD]:
    print "Processing slice %d" % i
    d_out = de_flicker(slc_window, s_filter, t_filter)
    out = d_out.cpu()
    cv2.imwrite(OUTPUT_FILE_PATH % (DEVICE, i), out)
    del slc_window[0]
    if i >= SLICE_RANGE[-MEDIAN_FILTER_RAD - 1]:
        slc_window.append(slc_window[2])
    else:
        slc_window.append(pp(im_read(SLICE_RANGE[i + MEDIAN_FILTER_RAD]).to(DEVICE)))
profiler.disable()
profiler.print_stats(1)
profiler.dump_stats(PROFILER_OUTPUT_PATH % (DEVICE, SLICE_RANGE[0]))
