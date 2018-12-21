import sys
import os
import numpy as np
import em_pre
import cv2
from skimage import measure
from scipy import ndimage
import shutil
import itertools
import cv2
from skimage.measure import label
import sys
import numpy as np
from itertools import product
from os import path, makedirs
from em_tools import getter

IDX_FILE_PATH = "/n/coxfs01/donglai/ppl/matin/Projects/EM-tools/script/slice.txt"
DFKR_SLICE_PATH = "/n/coxfs01/donglai/ppl/matin/cerebellum_P0_deflicker/%04d.png"
SLICE_NAME = "%04d.png"
OUTPUT_SLICE_DIR = "/n/coxfs01/donglai/ppl/matin/P0_final/"
BAD_MASK_SLICE_PATH = "/n/coxfs01/donglai/ppl/matin/bad_mask_P0/%04d.png"
TILE_RES = (1024, 1024)
X_RANGE = range(1, 13)
Y_RANGE = range(1, 18)
STEP = 10
THRESHOLD = 20
idx_array = np.loadtxt(IDX_FILE_PATH, delimiter='\n', dtype=int)
idx_array_len = len(idx_array)
dfkr_slice = getter.NormalSliceGetter(DFKR_SLICE_PATH)
mask = getter.NormalSliceGetter(BAD_MASK_SLICE_PATH)
output = getter.NormalSliceSetter(OUTPUT_SLICE_DIR, SLICE_NAME)


def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_dfkr_tile(idx, x_tile, y_tile):
    slc = dfkr_slice[idx]
    x_cor = x_tile - T_st[0]
    y_cor = y_tile - T_st[1]
    x_slc = np.s_[TILE_RES[0] * x_cor: TILE_RES[0] * (x_cor + 1)]
    y_slc = np.s_[TILE_RES[1] * y_cor: TILE_RES[1] * (y_cor + 1)]
    return slc[x_slc, y_slc]


def tile_getter(im, xId, yId):
    x_cor = xId - T_st[0]
    y_cor = yId - T_st[1]
    x_slc = np.s_[TILE_RES[0] * x_cor: TILE_RES[0] * (x_cor + 1)]
    y_slc = np.s_[TILE_RES[1] * y_cor: TILE_RES[1] * (y_cor + 1)]
    return im[x_slc, y_slc]


def get_tile_for_loop(slc, xId, yId, padSize=[0, 0], xRan=[1, 6], yRan=[1, 6]):
    im0 = tile_getter(slc, xId, yId)
    if min(padSize) == 0:
        return im0
    else:
        sz = im0.shape
        out = np.zeros((sz[0] + 2 * padSize[0], sz[1] + 2 * padSize[1]), dtype=np.uint8)
        out[padSize[0]:-padSize[0], padSize[1]:-padSize[1]] = im0
        # load up
        if xId == xRan[0]:  # reflect
            tmpSlice = im0[padSize[0]:0:-1]
        else:
            tmpSlice = tile_getter(slc, xId - 1, yId)[-padSize[0]:]
        out[:padSize[0], padSize[1]:-padSize[1]] = tmpSlice.copy()
        # load down
        if xId == xRan[1]:
            tmpSlice = im0[-2:-padSize[0] - 2:-1]
        else:
            tmpSlice = tile_getter(slc, xId + 1, yId)[:padSize[0]]
        out[-padSize[0]:, padSize[1]:-padSize[1]] = tmpSlice.copy()
        # load left
        if yId == yRan[0]:
            tmpSlice = out[:, 2 * padSize[0]:padSize[0]:-1]
        else:
            # left-middle
            tmpM = tile_getter(slc, xId, yId - 1)[:, -padSize[0]:]
            # left-up
            if xId == xRan[0]:
                tmpU = tmpM[padSize[0]:0:-1]
            else:
                tmpU = tile_getter(slc, xId - 1, yId - 1)[-padSize[0]:, -padSize[0]:]
            # left-down
            if xId == xRan[1]:
                tmpL = tmpM[-2:-padSize[0] - 2:-1]
            else:
                tmpL = tile_getter(slc, xId + 1, yId - 1)[:padSize[0], -padSize[0]:]
            tmpSlice = np.vstack((tmpU, tmpM, tmpL))
        out[:, :padSize[1]] = tmpSlice.copy()
        # load right
        if yId == yRan[1]:
            tmpSlice = out[:, -1 - padSize[0]:-1 - 2 * padSize[0]:-1]
        else:
            # right-middle
            tmpM = tile_getter(slc, xId, yId + 1)[:, :padSize[0]]
            # right-up
            if xId == xRan[0]:
                tmpU = tmpM[padSize[0]:0:-1]
            else:
                tmpU = tile_getter(slc, xId - 1, yId + 1)[-padSize[0]:, :padSize[0]]
            # right-down
            if xId == xRan[1]:
                tmpL = tmpM[-2:-padSize[0] - 2:-1]
            else:
                tmpL = tile_getter(slc, xId + 1, yId + 1)[:padSize[0], :padSize[0]]
            tmpSlice = np.vstack((tmpU, tmpM, tmpL))
        out[:, -padSize[1]:] = tmpSlice.copy()
        return out


def get_tile(idx, xId, yId, padSize=[0, 0], xRan=[1, 6], yRan=[1, 6]):
    im0 = get_dfkr_tile(idx, xId, yId)
    if min(padSize) == 0:
        return im0
    else:
        sz = im0.shape
        out = np.zeros((sz[0] + 2 * padSize[0], sz[1] + 2 * padSize[1]), dtype=np.uint8)
        out[padSize[0]:-padSize[0], padSize[1]:-padSize[1]] = im0
        # load up
        if xId == xRan[0]:  # reflect
            tmpSlice = im0[padSize[0]:0:-1]
        else:
            tmpSlice = get_dfkr_tile(idx, xId - 1, yId)[-padSize[0]:]
        out[:padSize[0], padSize[1]:-padSize[1]] = tmpSlice.copy()
        # load down
        if xId == xRan[1]:
            tmpSlice = im0[-2:-padSize[0] - 2:-1]
        else:
            tmpSlice = get_dfkr_tile(idx, xId + 1, yId)[:padSize[0]]
        out[-padSize[0]:, padSize[1]:-padSize[1]] = tmpSlice.copy()
        # load left
        if yId == yRan[0]:
            tmpSlice = out[:, 2 * padSize[0]:padSize[0]:-1]
        else:
            # left-middle
            tmpM = get_dfkr_tile(idx, xId, yId - 1)[:, -padSize[0]:]
            # left-up
            if xId == xRan[0]:
                tmpU = tmpM[padSize[0]:0:-1]
            else:
                tmpU = get_dfkr_tile(idx, xId - 1, yId - 1)[-padSize[0]:, -padSize[0]:]
            # left-down
            if xId == xRan[1]:
                tmpL = tmpM[-2:-padSize[0] - 2:-1]
            else:
                tmpL = get_dfkr_tile(idx, xId + 1, yId - 1)[:padSize[0], -padSize[0]:]
            tmpSlice = np.vstack((tmpU, tmpM, tmpL))
        out[:, :padSize[1]] = tmpSlice.copy()
        # load right
        if yId == yRan[1]:
            tmpSlice = out[:, -1 - padSize[0]:-1 - 2 * padSize[0]:-1]
        else:
            # right-middle
            tmpM = get_dfkr_tile(idx, xId, yId + 1)[:, :padSize[0]]
            # right-up
            if xId == xRan[0]:
                tmpU = tmpM[padSize[0]:0:-1]
            else:
                tmpU = get_dfkr_tile(idx, xId - 1, yId + 1)[-padSize[0]:, :padSize[0]]
            # right-down
            if xId == xRan[1]:
                tmpL = tmpM[-2:-padSize[0] - 2:-1]
            else:
                tmpL = get_dfkr_tile(idx, xId + 1, yId + 1)[:padSize[0], :padSize[0]]
            tmpSlice = np.vstack((tmpU, tmpM, tmpL))
        out[:, -padSize[1]:] = tmpSlice.copy()
        return out


def get_bb(seg):
    dim = len(seg.shape)
    a = np.where(seg > 0)
    if len(a) == 0:
        return [-1] * dim * 2
    out = []
    for i in range(dim):
        out += [a[i].min(), a[i].max()]
    return out


def get_patch(mat, xId, yId, padSize=[0, 0]):
    sz = mat.shape
    # abs: takes care of minus
    xRan = np.abs(np.arange(xId[0] - padSize[0], xId[1] + padSize[0]))
    yRan = np.abs(np.arange(yId[0] - padSize[1], yId[1] + padSize[1]))
    if xRan[-1] >= sz[0]:
        num = xRan[-1] - sz[0] + 1
        xRan[-num:] = np.arange(sz[0] - 2, sz[0] - 2 - num, -1)
    if yRan[-1] >= sz[1]:
        num = yRan[-1] - sz[1] + 1
        yRan[-num:] = np.arange(sz[1] - 2, sz[1] - 2 - num, -1)
    return mat[np.ix_(xRan, yRan)]


jobId = int(sys.argv[1])
jobNum = int(sys.argv[2])
sz_flo = 750
alpha = 0.01
ratio = 0.75
min_width = 64
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 1
medfilt_hsz = 2

WINDOW_RAD = 3
min_sz = 2
numT = [12, 18]
T_st = [1, 1]

kk = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kk2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# for 8nm/pix
r_if = 5  # image_sz/flow_sz
r_im = 10  # image_sz/mask_sz
pad_ip = 10  # pad for flow estimation (32nm)
pad_ip2 = r_if * pad_ip  # pad for interpolation (8nm)
bad_thres = 0.8  # clone can be weird without border
for i in range(jobId, idx_array_len, jobNum):
    # find nearby slices
    real_idx = idx_array[i]
    im_ids = [(j, idx_array[j]) for j in range(i - WINDOW_RAD, i + WINDOW_RAD + 1) if j >= 0 \
              and j <= idx_array_len - 1 and (real_idx - idx_array[j] ** 2 <= WINDOW_RAD ** 2)]
    less_ids = im_ids[:im_ids.index((i, real_idx))]
    greater_ids = im_ids[im_ids.index((i, real_idx)) + 1:]
    kk_iter = 4 if i in (0, idx_array_len - 1) else 2
    # 1. get full mask at 32nm
    mm_i = mask[i] / 255.0
    # mm_i = cv2.resize(cur_mask/255.0, None, fx=r_im/r_if, fy=r_im/r_if, interpolation=cv2.INTER_NEAREST)
    # write_tile(OUTPUT_SLICE_DIR, "Big_mask_before_dilation.png", 255 * mm_i)
    mm_i = (mm_i > 0).astype(np.uint8)
    mm_i = cv2.dilate(mm_i, kk, iterations=kk_iter)
    # write_tile(OUTPUT_SLICE_DIR, "Big_mask_0.png", 255 * mm_i.astype(np.uint8))
    mm_i = ndimage.morphology.binary_fill_holes(mm_i)
    # write_tile(OUTPUT_SLICE_DIR, "Big_mask.png", 255 * mm_i.astype(np.uint8))
    mm_left = mm_i.copy()
    print
    mm_i.max()
    if mm_i.max() == 0:
        # ideally: ip_bad + ip (copy everything)
        os.system('ln -s ' + DFKR_SLICE_PATH % (i + 1) + ' ' + OUTPUT_SLICE_DIR + SLICE_NAME % (i + 1))
        print
        "Created hard copy for %d." % (i + 1)
    out = dfkr_slice[i]
    if len(less_ids) == 0:
        less_ids.append((i, real_idx))
    if len(greater_ids) == 0:
        greater_ids.append((i, real_idx))
    for i0, i2 in sorted(itertools.product(less_ids, greater_ids), key=lambda x: (x[0][1] - x[1][1]) ** 2):
        warp_step = float(real_idx - i2[1]) / float(i0[1] - i2[1])
        print
        "interp %s with %s-%s at step %.2f" % ((i, real_idx), i0, i2, warp_step)
        mm_i = mm_left.copy()
        for k in [i0[0], i2[0]]:
            mm2 = mask[k] // 255
            mm2_i = mm2
            # mm2_i = cv2.resize(mm2//255, None,fx=r_im/r_if,fy=r_im/r_if, interpolation=cv2.INTER_NEAREST)
            mm2_i = cv2.dilate(mm2_i, kk, iterations=1)
            mm2_i = ndimage.morphology.binary_fill_holes(mm2_i)
            mm_i = np.logical_and(mm_i, mm2_i == False)
            mm_i = mask[k] / 255.0
            # cv2.resize(cur_mask/255.0, None, fx=r_im/r_if, fy=r_im/r_if, interpolation=cv2.INTER_NEAREST)
            # write_tile(OUTPUT_SLICE_DIR, "Big_mask_before_dilation.png", 255 * mm_i)
            mm_i = (mm_i > 0).astype(np.uint8)
            mm_i = cv2.dilate(mm_i, kk, iterations=kk_iter)
            # write_tile(OUTPUT_SLICE_DIR, "Big_mask_0.png", 255 * mm_i.astype(np.uint8))
            mm_i = ndimage.morphology.binary_fill_holes(mm_i)
            # write_tile(OUTPUT_SLICE_DIR, "Big_mask.png", 255 * mm_i.astype(np.uint8))
            # mm2_i = cv2.resize(mm2/255.0, None,fx=r_im/r_if,fy=r_im/r_if, interpolation=cv2.INTER_NEAREST)
            mm2_i = (mm2_i > 0).astype(np.uint8)
            mm2_i = cv2.dilate(mm2_i, kk2, iterations=kk_iter)
            # write_tile(OUTPUT_SLICE_DIR, "before box before fill.png", 255 * mm2_i.astype(np.uint8))
            mm2_i = ndimage.morphology.binary_fill_holes(mm2_i)
            # write_tile(OUTPUT_SLICE_DIR, "before box after fill.png", 255 * mm2_i.astype(np.uint8))
            mm_i = np.logical_and(mm_i, mm2_i == False)
            # write_tile(OUTPUT_SLICE_DIR, "before box after logic.png", 255 * mm_i.astype(np.uint8))
            mm_i = ndimage.morphology.binary_fill_holes(mm_i)
            cv2.imwrite(OUTPUT_SLICE_DIR + "before box.png", 255 * mm_i.astype(np.uint8))
            print
            mm_i.max()
        print
        mm_i.shape
        if mm_i.max():
            # need to overwrite
            for j in range(numT[0]):
                for k in range(numT[1]):
                    mm_i2 = mm_i[sz_flo * j: sz_flo * (j + 1), sz_flo * k: sz_flo * (k + 1)]
                    print
                    mm_i2.max()
                    if mm_i2.max():  # need to interp
                        print
                        'do: ', i, i0, i2, j, k
                        # dilate a bit
                        mm_i2 = np.pad(cv2.dilate(mm_i2.astype(np.uint8), kk2, iterations=kk_iter), pad_ip, 'constant')
                        # for inpaint
                        im0_b = get_tile(i0[0], j + T_st[0], k + T_st[1], [pad_ip * r_if, pad_ip * r_if],
                                         xRan=[T_st[0], T_st[0] + numT[0] - 1],
                                         yRan=[T_st[1], T_st[1] + numT[1] - 1])
                        im2_b = get_tile(i2[0], j + T_st[0], k + T_st[1], [pad_ip * r_if, pad_ip * r_if],
                                         xRan=[T_st[0], T_st[0] + numT[0] - 1],
                                         yRan=[T_st[1], T_st[1] + numT[1] - 1])
                        # for flow
                        im0 = cv2.resize(im0_b, None, fx=1.0 / r_if, fy=1.0 / r_if, interpolation=cv2.INTER_LINEAR)
                        im2 = cv2.resize(im2_b, None, fx=1.0 / r_if, fy=1.0 / r_if, interpolation=cv2.INTER_LINEAR)
                        u, v, _ = em_pre.coarse2fine_flow(im0[:, :, None].astype(float) / 255.0,
                                                          im2[:, :, None].astype(float) / 255.0,
                                                          -1, medfilt_hsz,
                                                          alpha, ratio, min_width, nOuterFPIterations,
                                                          nInnerFPIterations,
                                                          nSORIterations, colType)
                        print
                        "u", u.max(), u.min(), u.mean(), "v", v.max(), v.min(), v.mean()
                        if np.count_nonzero(mm_i2) / float(np.prod(mm_i2.shape)) > bad_thres:
                            print
                            'direct interp'
                            flow = np.stack([cv2.resize(u, None, fx=r_if, fy=r_if, interpolation=cv2.INTER_LINEAR), \
                                             cv2.resize(v, None, fx=r_if, fy=r_if, interpolation=cv2.INTER_LINEAR)],
                                            axis=2).astype(np.float32) * r_if * warp_step
                            im1_b = em_pre.warpback_image(im2_b, flow, opt_interp=1, opt_border=1)[:, :, None]
                        else:
                            print
                            "indirect interp"
                            im1_b = get_tile_for_loop(out, j + T_st[0], k + T_st[1], [pad_ip * r_if, pad_ip * r_if],
                                                      xRan=[T_st[0], T_st[0] + numT[0] - 1],
                                                      yRan=[T_st[1], T_st[1] + numT[1] - 1])
                            print
                            im1_b.shape
                            # mask out image data with gray
                            tmp_m = cv2.resize(
                                get_patch(mm_i, [sz_flo * j, sz_flo * (j + 1)], [sz_flo * k, sz_flo * (k + 1)],
                                          [pad_ip, pad_ip]).astype(np.uint8), None,
                                fx=r_if, fy=r_if, interpolation=cv2.INTER_NEAREST)
                            print
                            mm_i.shape
                            # tmp_m = get_patch(mm_i, [sz_flo*j,sz_flo*(j+1)], [sz_flo*k,sz_flo*(k+1)], [pad_ip,pad_ip]).astype(np.uint8)
                            # tmp_m = cv2.resize(tmp_m, (1124, 1124), interpolation=cv2.INTER_NEAREST)
                            im1_b = im1_b * (tmp_m == 0).astype(np.uint8) + 150 * (tmp_m == 1).astype(np.uint8)
                            im1_b = np.tile(im1_b[:, :, None], [1, 1, 3])
                            # write_tile(OUTPUT_SLICE_DIR, "4_3_initial.png", im1_b)
                            bb = measure.label(mm_i2, background=0)
                            # write_tile(OUTPUT_SLICE_DIR, "4_3_mask.png", bb)
                            bb_l = np.delete(np.unique(bb), 0)
                            for ll in bb_l:
                                # for each region
                                print
                                '\t start clone:', ll
                                seg = (bb == ll).astype(np.uint8) * 255
                                box = get_bb(seg)
                                # make sure it's odd size
                                box[1] -= (box[1] - box[0] + 1) % 2
                                box[3] -= (box[3] - box[2] + 1) % 2
                                if (box[1] - box[0]) >= min_sz and (box[3] - box[2]) >= min_sz:
                                    flow = np.stack(
                                        [cv2.resize(u[box[0] - pad_ip:box[1] + pad_ip, box[2] - pad_ip:box[3] + pad_ip],
                                                    None, fx=r_if, fy=r_if, interpolation=cv2.INTER_LINEAR),
                                         cv2.resize(v[box[0] - pad_ip:box[1] + pad_ip, box[2] - pad_ip:box[3] + pad_ip],
                                                    None, fx=r_if, fy=r_if, interpolation=cv2.INTER_LINEAR)],
                                        axis=2).astype(np.float32) * r_if * warp_step
                                    box2 = [x * r_if for x in box]
                                    im_p = em_pre.warpback_image(
                                        im2_b[box2[0] - pad_ip2:box2[1] + pad_ip2, box2[2] - pad_ip2:box2[3] + pad_ip2],
                                        flow,
                                        opt_interp=1, opt_border=1)[pad_ip2:-pad_ip2, pad_ip2:-pad_ip2]
                                    # interp
                                    box2_c = tuple([(box2[2 * f + 1] + box2[2 * f]) // 2 for f in [1, 0]])

                                    im1_b = cv2.seamlessClone(np.tile(im_p[:, :, None], [1, 1, 3]), im1_b,
                                                              cv2.resize(seg[box[0]:box[1], box[2]:box[3]], None,
                                                                         fx=r_if, fy=r_if,
                                                                         interpolation=cv2.INTER_NEAREST),
                                                              box2_c, cv2.NORMAL_CLONE)
                        x_idx = np.s_[TILE_RES[0] * j: TILE_RES[0] * (j + 1)]
                        y_idx = np.s_[TILE_RES[1] * k: TILE_RES[1] * (k + 1)]
                        # write_tile(OUTPUT_SLICE_DIR, "4_3_output.png", im1_b)
                        out[x_idx, y_idx] = im1_b[pad_ip2:-pad_ip2, pad_ip2:-pad_ip2, 0]
            if mm_left.max() == 0:
                break
    print
    'done', OUTPUT_SLICE_DIR, SLICE_NAME % i
    output[i] = out

