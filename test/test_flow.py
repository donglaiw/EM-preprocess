import unittest 

import sys
import numpy as np
import time
import imageio

import em_pre

def getParam():
    alpha = 0.01
    ratio = 0.75
    minWidth = 64
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    warp_step = 1
    medfilt_hsz = 2
    return alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, \
            nSORIterations, colType, warp_step, medfilt_hsz

def getData():
    import h5py
    # input data
    DD='/n/coxfs01/donglai/data/Alex/ROI215/'
    chunk='2_2'
    D1 = DD + 'df150_ds_h5_ind_slope/'
    img = np.array(h5py.File(D1+chunk+'/'+chunk+'_0_1.h5')['main'])
    for i in range(5):
        imageio.imwrite('test/im'+str(i)+'.png', img[:,:,i])

def check_warp_mf():
    im1 = imageio.imread('test/im1.png')[:,:,None].astype(float) / 255.
    im2 = imageio.imread('test/im2.png')[:,:,None].astype(float) / 255.
    ims = np.stack([im1,im2],axis=0)
    alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, \
        nSORIterations, colType, warp_step, medfilt_hsz = getParam()

    for warp_step in [0.5,1.0]:
        im2W2 = em_pre.coarse2fine_ims(
            ims, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType, warp_step, 1, 0)
        imageio.imwrite('test/im2W_'+str(warp_step)+'.png',np.uint8(np.clip(255*im2W2[0,:,:,0],0,255)))
        """
        im2W2m = em_pre.coarse2fine_ims(
            ims, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType, warp_step, 1, 2)
        imageio.imwrite('test/im2Wm_'+str(warp_step)+'.png',np.uint8(255*im2W2m[0,:,:,0]))
        """

class TestIDM(unittest.TestCase):
    def test_warp(self):
        im1 = imageio.imread('test/im1.png')[:,:,None].astype(float) / 255.
        im2 = imageio.imread('test/im2.png')[:,:,None].astype(float) / 255.

        alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, \
            nSORIterations, colType, warp_step, medfilt_hsz = getParam()
        minWidth = 64

        u, v, im2W = em_pre.pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType, warp_step, medfilt_hsz)

        ims = np.stack([im1,im2],axis=0)
        im2W2 = em_pre.coarse2fine_ims(
            ims, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
            nSORIterations, colType, warp_step, 1, medfilt_hsz)

        self.assertTrue(np.max(np.abs(im2W-im2W2[0]))<1e-4)



if __name__ == '__main__':                                                                                                           
    #getData()
    check_warp_mf()
    #unittest.main()
