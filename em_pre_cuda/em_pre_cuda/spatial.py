import torch
from torch.nn.functional import conv2d, pad


class PyTorch2dConvolution:
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
