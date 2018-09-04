import torch
import em_pre_cuda.deflicker as dfkr_gpu
import cv2
import cProfile
import numpy as np
from itertools import product
from os import path

PROF_DIR = "./Profiler_Results/deflicker_all_cerebellum.profile"
IDX_FILE = "/n/coxfs01/donglai/data/cerebellum/data/unique_slice_v1.txt"
INPUT_DIR = "/n/coxfs01/donglai/data/cerebellum/orig_png/%04d/"
TILE_NAME = "%d_%d.png"
OUTPUT_DIR = "./n/coxfs01/donglai/ppl/matin/cerebellum_df/%d.png"
NUM_SLICE = 2513
TILE_DIM = (3750, 3750)
ROW_SZ = 3
COL_SZ = 3
ROW_IDX_0 = 2
COL_IDX_0 = 3

def read_tile(path):
    """
    CV2 wrapper for reading tiles. Here to reduce bugs in my code.
    :param path: Path to the tile.
    :return: The requested tile in Gray scale, with dtype=np.float32.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def replace_missing(missing_idx, tile_name):
    """
    Replaces a missing tile by replacing it with an existing tile from earlier indices.
    :param missing_idx: the index of the slice with the missing tile.
    :param tile_name: Name of the tile on disk e.g. 4_3.png.
    :return: The first existing tile before the missing index.
    :raises: IndexError if no tiles were found before the missing tile to replace the current tile (Highly unlikely).
    """
    current_idx = missing_idx - 1
    while True:
        curr_tile_path = (INPUT_DIR % (current_idx + 1)) + tile_name
        if path.exists(curr_tile_path):
            return read_tile(current_idx)
        current_idx = current_idx - 1
        if current_idx < 0:
            raise IndexError("Failed to recover the missing tile for %s" % tile_name)


def deflicker_cerebellum():
    idx_array = open(IDX_FILE).readlines()
    profiler = cProfile.Profile()

    def get_slice(idx):
        """
        Reconstructs a slice from its smaller tiles. Replaces missing tiles with their previously existing ones.
        :param idx: index of the slice
        :return: The desired slice as a Torch CUDA Tensor.
        """
        slice_dir = INPUT_DIR % int(idx_array[idx])
        slice_cpu = np.zeros(TILE_DIM[0] * ROW_SZ, TILE_DIM[1] * COL_SZ)
        for i, j in product(range(ROW_SZ), range(COL_SZ)):
            cur_tile_name = TILE_NAME % (i + ROW_IDX_0, j + COL_IDX_0)
            cur_tile_path = slice_dir + cur_tile_name
            cur_tile = read_tile(cur_tile_path) if path.exists(cur_tile_path) \
                else replace_missing(idx, cur_tile_name)
            cur_tile_x_idx = np.s_[TILE_DIM[0] * i: TILE_DIM[0] * (i + 1)]
            cur_tile_y_idx = np.s_[TILE_DIM[1] * j: TILE_DIM[1] * (j + 1)]
            slice_cpu[cur_tile_x_idx, cur_tile_y_idx] = cur_tile
        slice_gpu = torch.from_numpy(slice_cpu).cuda()
        del slice_cpu
        return slice_gpu

    profiler.enable()
    dfkr_gpu.deflicker_online(get_slice, len(idx_array),
                              global_stat=(150, -1),
                              pre_proc_method='threshold',
                              s_flt_rad=15,
                              t_flt_rad=2,
                              write_dir=OUTPUT_DIR)
    profiler.disable()
    torch.cuda.empty_cache()
    profiler.print_stats(sort=1)
    profiler.dump_stats(PROF_DIR)


if __name__ == "__main__":
    deflicker_cerebellum()
