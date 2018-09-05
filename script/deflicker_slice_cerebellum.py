import torch
import em_pre_cuda.deflicker as dfkr_gpu
import cv2
import cProfile
import sys
import numpy as np
from itertools import product
from os import path


# PROF_RESULT_PATH = "./Profiler_Results/deflicker_all_cerebellum.profile"
IDX_FILE_PATH = "/n/coxfs01/donglai/data/cerebellum/data/unique_slice_v1.txt"
INPUT_TILE_DIR = "/n/coxfs01/donglai/data/cerebellum/orig_png/%04d/"
TILE_NAME = "%d_%d.png"
OUTPUT_SLICE_PATH = "/n/coxfs01/donglai/ppl/matin/cerebellum_df/%04d.png"
NUM_SLICE = 2513
TILE_RES = (3750, 3750)
TILES_IN_ROW = 3
TILES_IN_COL = 3
ROW_IDX_0 = 2
COL_IDX_0 = 3

def read_tile(path):
    """
    CV2 wrapper for reading tiles. Here to reduce bugs in my code.
    :param path: Path to the tile.
    :return: The requested tile in Gray scale, with dtype=np.float32.
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def replace_missing_tile(missing_idx, tile_name):
    """
    Replaces a missing tile by replacing it with an existing tile from earlier indices.
    :param missing_idx: the index of the slice with the missing tile.
    :param tile_name: Name of the tile on disk e.g. 4_3.png.
    :return: The first existing tile before the missing index.
    :raises: IndexError if no tiles were found before the missing tile to replace the current tile (Highly unlikely).
    """
    current_idx = missing_idx - 1
    while True:
        curr_tile_path = (INPUT_TILE_DIR % (current_idx + 1)) + tile_name
        if path.exists(curr_tile_path):
            return read_tile(current_idx)
        current_idx = current_idx - 1
        if current_idx < 0:
            raise IndexError("Failed to recover the missing tile for %s" % tile_name)


def deflicker_cerebellum():
    idx_array = open(IDX_FILE_PATH).readlines()
    profiler = cProfile.Profile()
    df_range = range(int(sys.argv[1]), int(sys.argv[2]))


    def get_slice(idx):
        """
        Reconstructs a slice from its smaller tiles. Replaces missing tiles with their previously existing ones.
        :param idx: index of the slice
        :return: The desired slice as a Torch CUDA Tensor.
        """
        slice_dir = INPUT_TILE_DIR % int(idx_array[idx])
        slice_cpu = np.zeros((TILE_RES[0] * TILES_IN_ROW, TILE_RES[1] * TILES_IN_COL), dtype=np.float32)
        for i, j in product(range(TILES_IN_ROW), range(TILES_IN_COL)):
            cur_tile_name = TILE_NAME % (j + COL_IDX_0, i + ROW_IDX_0)
            cur_tile_path = slice_dir + cur_tile_name
            cur_tile = read_tile(cur_tile_path) if path.exists(cur_tile_path) \
                else replace_missing_tile(idx, cur_tile_name)
            cur_tile_x_idx = np.s_[TILE_RES[0] * i: TILE_RES[0] * (i + 1)]
            cur_tile_y_idx = np.s_[TILE_RES[1] * j: TILE_RES[1] * (j + 1)]
            slice_cpu[cur_tile_x_idx, cur_tile_y_idx] = cur_tile
        slice_gpu = torch.from_numpy(slice_cpu).cuda()
        del slice_cpu
        return slice_gpu

    profiler.enable()
    dfkr_gpu.deflicker_online(get_slice, df_range,
                              global_stat=(150, -1),
                              pre_proc_method='threshold',
                              s_flt_rad=15,
                              t_flt_rad=2,
                              write_dir=OUTPUT_SLICE_PATH)
    profiler.disable()
    torch.cuda.empty_cache()
    profiler.print_stats(sort=1)
    # profiler.dump_stats(PROF_RESULT_PATH)


if __name__ == "__main__":
    deflicker_cerebellum()
