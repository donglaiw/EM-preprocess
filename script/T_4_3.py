import torch
import em_pre_cuda.deflicker as dfkr_gpu
import cv2
import cProfile

IMG_DIR = "/n/coxfs01/donglai/data/cerebellum/orig_png/%04d/4_3.png"
IDX_FILE = "/n/coxfs01/donglai/data/cerebellum/data/unique_slice_4_3.txt"
OUT_DIR = "/n/coxfs01/donglai/data/cerebellum/df150_png/%04d/4_3_gpu.png"
NUM_SLICE = 2513

def deflicker_4_3():
    # online version
    idx_array = open(IDX_FILE).readlines()
    gpu_profile = cProfile.Profile()

    def get_im(idx):
        im_np = cv2.imread(IMG_DIR % int(idx_array[idx]))
        return torch.from_numpy(im_np).cuda()

    gpu_profile.enable()
    dfkr_gpu.deflicker_online(get_im, opts=(1, 0, 0), global_stat=(150, -1), s_flt_rad=15, t_flt_rad=2,
                              write_dir=OUT_DIR)
    gpu_profile.disable()
    torch.cuda.empty_cache()
    gpu_profile.print_stats(sort=1)
    gpu_profile.dump_stats("gpu.profile")


if __name__ == "__main__":
    deflicker_4_3()
