""" Pytorch custom transforms
"""

import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor:torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.shape) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean:.2f}, std={self.std:.2f})'
