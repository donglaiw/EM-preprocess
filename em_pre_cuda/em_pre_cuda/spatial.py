import torch
from torch.nn.functional import conv2d, pad


class PyTorch2dConvolution:
    def __init__(self, radius, device):
        diam = 2 * radius + 1
        sqz_kernel = torch.ones((diam,) * 2, device=device).div_(diam ** 2)
        self.kernel = torch.unsqueeze(torch.unsqueeze(sqz_kernel, 0), 0)
        self.padding = (radius,) * 4

    def __call__(self, image):
        unsqz_im = torch.unsqueeze(torch.unsqueeze(image, 0), 0)
        pad_img = pad(unsqz_im, self.padding, mode='reflect')
        output = conv2d(pad_img, self.conv_kernel)
        return torch.squeeze(output)
