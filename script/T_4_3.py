import torch
import em_pre_cuda.deflicker as dfkr_gpu
import cv2
import cProfile
import numpy as np

IMG_DIR = "/n/coxfs01/donglai/data/cerebellum/orig_png/%04d/4_3.png"
IDX_FILE = "/n/coxfs01/donglai/data/cerebellum/data/unique_slice_4_3.txt"
OUT_DIR = "%04d/4_3_gpu.png"
NUM_SLICE = 2513

def deflicker_4_3():
    # online version
    idx_array = open(IDX_FILE).readlines()
    gpu_profile = cProfile.Profile()

    def get_im(idx):
        im_dir = IMG_DIR % int(idx_array[idx])
        im_np = cv2.imread(im_dir, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        output = torch.from_numpy(im_np).cuda()
        return output

    gpu_profile.enable()
    dfkr_gpu.deflicker_online(get_im, len(idx_array),
                              global_stat=(150, -1),
                              pre_proc_method='threshold',
                              s_flt_rad=15,
                              t_flt_rad=2,
                              write_dir=OUT_DIR)
    gpu_profile.disable()
    torch.cuda.empty_cache()
    gpu_profile.print_stats(sort=1)
    gpu_profile.dump_stats("gpu.profile")


if __name__ == "__main__":
    deflicker_4_3()
