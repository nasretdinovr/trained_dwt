import math

import torch
from torch import nn
import torch.nn.functional as F
from thresholds.Threshold_base import ThresholdBase

class ThresholdDWT(ThresholdBase):
    """Class for speech denoising with using learnable discrete wavelet transform
            Args:
            threshold (flat): threshold value

            requires_grad (bool): whether train wavelet filters or not

            thresholding_algorithm (string): thresholding algorithm used on wavelet decomposition of noise speech.
            Can be hard or soft. <https://ieeexplore.ieee.org/document/7455802>

            threshold_mode (string):
            "level-dependent": usage different thresholding values for each wavelet decomposition level
            "global" - usage single thresholding values for all wavelet decomposition level

            signal_length (int): length of input signal in points

            num_wavelet_levels (int): number of decomposition levels of wavelet transform

            sigma (float): noise variance for using in baseline model

            thresholding_parameter (float): thresholding function parameter. If 0, then ether hard or soft
            thresholding functions became vanilla.
            For more information - https://pdfs.semanticscholar.org/f81f/a9ab84ddad5e8f8730b4ed7a0879924666b2.pdf
        """

    def __init__(self, threshold=0.05,
                 requires_grad=True,
                 thresholding_algorithm='hard',
                 mode='global',
                 signal_length=None,
                 num_wavelet_levels=None,
                 sigma=None,
                 thresholding_parameter=0.05):
        super(ThresholdDWT, self).__init__(threshold, requires_grad, thresholding_algorithm, mode, signal_length,
                                                 num_wavelet_levels, sigma, thresholding_parameter)
        if mode == 'level_dependent':
            assert signal_length is not None, "level_dependent mode requires: signal_length"
            assert num_wavelet_levels is not None, "level_dependent mode requires: num_wavelet_levels"

            if self.signal_length % 2 ** self.num_wavelet_levels != 0:
                self.signal_length += 2 ** self.num_wavelet_levels - (self.signal_length % 2 ** self.num_wavelet_levels)

            threshold = [self.compute_level_threshold(i) for i in range(self.num_wavelet_levels)]
            threshold.append(self.compute_level_threshold(self.num_wavelet_levels - 1))
            self.threshold = nn.Parameter(torch.tensor(threshold), requires_grad=self.requires_grad)

    def level_dependent_threshold(self, signal):
        start = 0
        length = signal.size(2)
        for i in range(self.num_wavelet_levels):
            length = length // 2
            lvl_slice = signal[:, :, start:start + length].clone()
            signal[:, :, start:start + length] = getattr(self, self.thresholding_algorithm)(lvl_slice,
                                                                                            self.threshold[i])
            start += length
        signal[:, :, start:] = getattr(self, self.thresholding_algorithm)(signal[:, :,  start:].clone(),
                                                                          self.threshold[self.num_wavelet_levels])
        return signal

    def forward(self, x):
        if self.mode == 'level_dependent':
            return self.level_dependent_threshold(x)
        else:
            return getattr(self, self.thresholding_algorithm)(x, self.threshold)

