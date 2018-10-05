"""****************************************************************************
* Name: T_deflicker.py
* Author: Matin Raayai Ardakani, Donglai Wei
* A Benchmark/Test run for comparing the performance of the em_pre and 
* em_pre_cuda deflicker function. 
****************************************************************************"""
import cProfile
import h5py
import torch
import numpy as np
import em_pre.deflicker as dfkr_cpu
import em_pre_cuda.deflicker as dfkr_gpu
from T_util import writeh5

IN_DATA_PATH = "/home/matinraayai/Desktop/coxfs01_matin/data_h5/cerebellum_ds_orig.h5"
INPUT_DATASET = "main"
OUTPUT_DATASET = "main"
CPU_IM_OUT_PATH = "../tmp/T_deflicker_cpu_out.h5"
GPU_IM_OUT_PATH = "../tmp/T_deflicker_gpu_out.h5"
CPU_PROF_OUT_PATH = "../tmp/cpu.profile"
GPU_PROF_OUT_PATH = "../tmp/gpu.profile"


def read_h5(data_path, dataset_name):
    return h5py.File(data_path, 'r')[dataset_name]


def test_snemi():
    # load data: 100x1024x1024
    ims_np = read_h5(IN_DATA_PATH, INPUT_DATASET)
    cpu_profile = gpu_profile = cProfile.Profile()

    def get_n_np(idx):
        """Getter for em_pre.deflicker."""
        return np.array(ims_np[idx], dtype=np.float32)

    def get_n_cuda(idx):
        """Getter fir em_pre_cuda.deflicker."""
        return torch.tensor(ims_np[idx], dtype=torch.float32).cuda()

    # em_pre.deflicker test:
    cpu_profile.enable()
    out_cpu = dfkr_cpu.deflicker_online(get_n_np, 
                                        opts=[0, 0, 0],
                                        globalStat=[150, -1], 
                                        filterS_hsz=[15, 15],
                                        filterT_hsz=2,
                                        output="cpu_%d.png")
    cpu_profile.disable()
    # em_pre_cuda.deflicker test:
    gpu_profile.enable()
    out_gpu = dfkr_gpu.deflicker_online(get_n_cuda, slice_range=range(0,2020),
                                        pre_proc_method='naive',
                                        global_stat=(150, -1), 
                                        s_flt_rad=15, 
                                        t_flt_rad=2,
                                        write_dir="gpu_%d.png")
    gpu_profile.disable();torch.cuda.empty_cache()
    
    writeh5(CPU_IM_OUT_PATH, OUTPUT_DATASET, out_cpu)
    writeh5(GPU_IM_OUT_PATH, OUTPUT_DATASET, out_gpu)
    
    cpu_profile.print_stats(sort=1); gpu_profile.print_stats(sort=1)
    cpu_profile.dump_stats(CPU_PROF_OUT_PATH)
    gpu_profile.dump_stats(GPU_PROF_OUT_PATH)


if __name__ == "__main__":
    test_snemi()
