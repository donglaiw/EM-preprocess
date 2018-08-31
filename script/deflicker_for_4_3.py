"""****************************************************************************
* Name: T_deflicker.py
* Author: Matin Raayai Ardakani, Donglai Wei
* A Benchmark/Test run for comparing the performance of the em_pre and em_pre_cuda
* deflicker function. 
****************************************************************************"""
import h5py
import torch
import numpy as np
import em_pre_cuda.deflicker as dfkr_gpu
from T_util import writeh5

IN_DATA_PATH = "/mnt/d5d402f1-882e-4686-99db-77f08c51ac84/Data/4_3_downsampled_debug.h5"
INPUT_DATASET = "main"
OUTPUT_DATASET = "main"
CPU_IM_OUT_PATH = "./test_output/T_deflicker_cpu_out.h5"
GPU_IM_OUT_PATH = "./test_output/T_deflicker_gpu_out.h5"
CPU_PROF_OUT_PATH = "./cpu.profile"
GPU_PROF_OUT_PATH = "./gpu.profile"


def read_h5_as_np(data_path, dataset_name):
    a= h5py.File(data_path, 'r')[dataset_name]
    return np.array(a, dtype=np.float32)


def df():
    
    ims_np = read_h5_as_np(IN_DATA_PATH, INPUT_DATASET)
    print ims_np.shape
    ims_torch = torch.from_numpy(ims_np.copy()).cuda()

    # def get_n_np(idx):
    #     """Getter for em_pre.deflicker."""
    #     return ims_np[idx]

    def get_n_cuda(idx):
        """Getter fir em_pre_cuda.deflicker."""
        return ims_torch[idx]

    # out_cpu = dfkr_cpu.deflicker_online(get_n_np,
    #                                     opts=[0, 0, 0],
    #                                     globalStat=[150, -1],
    #                                     filterS_hsz=[15, 15],
    #                                     filterT_hsz=2)

    out_gpu = dfkr_gpu.deflicker_online(get_n_cuda, num_slice=2057,
                                        global_stat=(150, -1),
                                        pre_proc_method='threshold',
                                        s_flt_rad=15, 
                                        t_flt_rad=2)
    torch.cuda.empty_cache()
    writeh5(GPU_IM_OUT_PATH, OUTPUT_DATASET, out_gpu)


if __name__ == "__main__":
    df()
