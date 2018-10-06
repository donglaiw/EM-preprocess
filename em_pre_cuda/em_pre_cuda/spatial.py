"""*********************************************************************************************************************
 * Name: spatial.py
 * Author(s): Donglai Wei, Matin Raayai Ardakani
 * Email(s): weiddoonngglai@gmail.com, raayai.matin@gmail.com
 * Possible spatial filters to be used in de_flicker. After initialization, these filters are callable with a single
 image as their argument.
 ********************************************************************************************************************"""
import torch
from torch.nn.functional import conv2d, pad
import cv2
import numpy as np

class PyTorch2dMean:
    """
    A mean filter implemented using Pytorch. Can work with both cpu and gpu Pytorch tensors.
    """
    def __init__(self, radius):
        diam = 2 * radius + 1
        sqz_kernel = torch.ones((diam,) * 2).div_(diam ** 2)
        self.kernel = torch.unsqueeze(torch.unsqueeze(sqz_kernel, 0), 0)
        self.padding = (radius,) * 4
        self.dvc_not_checked = True

    def __transfer_kernel(self, image):
        if self.dvc_not_checked:
            self.kernel = self.kernel.to(image.device)
            self.dvc_not_checked = False
    def __call__(self, image):
        self.__transfer_kernel(image)
        unsqz_im = torch.unsqueeze(torch.unsqueeze(image, 0), 0)
        pad_img = pad(unsqz_im, self.padding, mode='reflect')
        output = conv2d(pad_img, self.kernel)
        return torch.squeeze(output)

class cv2Mean:
    """
    A mean filter using cv2's filter 2d. Can only work with cpu Pytorch tensors.
    """
    def __init__(self, filter_rad):
        diam = 2 * filter_rad + 1
        self.filter = np.ones((diam,) * 2, dtype=np.float32) / (diam ** 2)

    def __Call__(self, ims):
        ims_numpy = ims.numpy()
        return torch.from_numpy(cv2.filter2D(ims_numpy, -1, self.filter, borderType=cv2.BORDER_REFLECT_101))
