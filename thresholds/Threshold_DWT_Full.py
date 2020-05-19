import math

import torch
from torch import nn
import torch.nn.functional as F
from thresholds.Threshold_base import ThresholdBase

class ThresholdDWTFull(ThresholdBase):
    """Class for speech denoising with using learnable discrete wavelet transform with full filterbank
            Args:
            threshold (flat): threshold value

            requires_grad (bool): whether train wavelet filters or not

            thresholding_algorithm (string): thresholding algorithm used on wavelet decomposition of noise speech.
            Can be hard or soft. <https://ieeexplore.ieee.org/document/7455802>

            threshold_mode (string):
            "level-dependent": usage different thresholding values for each wavelet decomposition level
            "global" - usage single thresholding values for all wavelet decomposition level

            signal_length (int): length of input signal in points

            num_wavelet_levels (int): decomposition levels number of wavelet transform

            sigma (float): noise variance used in baseline model

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
        super(ThresholdDWTFull, self).__init__(threshold, requires_grad, thresholding_algorithm, mode, signal_length,
                                                 num_wavelet_levels, sigma, thresholding_parameter)
        if mode == 'level_dependent':
            assert signal_length is not None, "level_dependent mode requires: signal_length"
            assert num_wavelet_levels is not None, "level_dependent mode requires: num_wavelet_levels"

            if self.signal_length % 2 ** self.num_wavelet_levels != 0:
                self.signal_length += 2 ** self.num_wavelet_levels - (self.signal_length % 2 ** self.num_wavelet_levels)
            threshold = self.compute_level_threshold(self.num_wavelet_levels - 1)
            self.threshold = nn.Parameter(torch.ones((2 ** self.num_wavelet_levels, 1))*threshold,
                                          requires_grad=self.requires_grad)

    def forward(self, x):
        if self.mode == 'level_dependent':
            x = x.reshape(x.size(0), x.size(1), 2 ** self.num_wavelet_levels, -1)
            x = getattr(self, self.thresholding_algorithm)(x, self.threshold)
            return x.reshape(x.size(0), x.size(1), -1)
        else:
            return getattr(self, self.thresholding_algorithm)(x, self.threshold)

