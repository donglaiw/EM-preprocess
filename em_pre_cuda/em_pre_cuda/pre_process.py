import torch


class NaivePreProcess:
    def __init__(self, global_stat=None):
        self.global_stat = global_stat

    def __call__(self, image):
        if self.global_stat is None:
            self.global_stat = (image.mean(), image.std())
        mm = image.mean(dim=0).mean(dim=0)
        if self.global_stat[1] > 0:
            tmp_std = image.std().item()
            if tmp_std < 1e-3:
                image.sub_(mm).add_(self.global_stat[0])
            else:
                image.sub_(mm).div_(tmp_std).mul_(self.global_stat[1]).add_(self.global_stat[0])
        else:
            if self.global_stat[0] > 0:
                image.sub_(mm).add_(self.global_stat[0])
        return image


class ThresholdPreProcess:
    def __init__(self, global_stat=None, sampling_step=10, mask_threshold=(10, 245)):
        self.global_stat = global_stat
        self.sampling_step = sampling_step
        self.mask_threshold = mask_threshold

    def __call__(self, image):
        if self.global_stat is None:
            self.global_stat = (image.mean(), image.std())
        im_copy = image[::self.sampling_step, ::self.sampling_step]
        if self.mask_threshold[0] is not None:
            im_copy = im_copy[im_copy > self.mask_threshold[0]]
        if self.mask_threshold[1] is not None:
            im_copy = im_copy[im_copy < self.mask_threshold[1]]
        image.sub_(im_copy.median()).add_(self.global_stat[0])
        if self.mask_threshold[0] is not None:  # for artifact/boundary
            image[image < self.mask_threshold[0]] = self.mask_threshold[0]
        if self.mask_threshold[1] is not None:  # for blood vessel
            image[image > self.mask_threshold[1]] = self.mask_threshold[1]
        return image
