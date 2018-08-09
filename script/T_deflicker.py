# test video deflicker
import h5py
import torch
import em_pre.deflicker as dfkr_cpu
import em_pre_cuda.deflicker as dfkr_gpu
from T_util import writeh5


def test_snemi():
    # load data: 100x1024x1024
    ims = torch.tensor(h5py.File('/n/coxfs01/donglai/data/SNEMI3D/train-input.h5')['main'], device='cuda')
    ims = ims.permute(1, 2, 0)

    # online version
    def getN(i):
        return ims[i]

    out_cpu = dfkr_cpu(getN, opts=[0, 0, 0], globalStat=[150, -1], filterS_hsz=[15,15], filterT_hsz=2)
    out_gpu = dfkr_gpu(getN, opts=[0, 0, 0], global_stat=(150, -1), s_flt_rad=(15, 15), t_flt_rad=2)
    writeh5('tmp/snemi_df150_online.h5', 'main', out)


if __name__ == "__main__":
    test_snemi()
