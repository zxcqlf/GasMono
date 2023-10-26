# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from .hrlayers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc = [64,128,216,288,288], scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.use_fs = True

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("upconv_edge", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("atten", i, 0)] = AttModule(num_ch_in, num_ch_in)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("upconv_edge", i, 1)] = ConvBlock(self.num_ch_dec[i], num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            if self.use_fs:
                self.convs[("mask", s)] = nn.Sequential(
                nn.Conv2d(self.num_ch_dec[s], self.num_ch_dec[s], 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.num_ch_dec[s], ((2**s)*(2**s))*9, 1, padding=0))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        y = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("atten", i, 0)](x,y)
            #====  disp part  ====
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            #====  edge part  ====
            y = self.convs[("upconv_edge", i, 0)](y)
            y = upsample(y)
            y = self.convs[("upconv_edge", i, 1)](y)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                if self.use_fs and i >0:
                    mask= .25* self.convs[("mask", i)](y)
                    self.outputs[("disp", i)] = upsample_disp(disp=self.outputs[("disp", i)],mask=mask,scale=i)
                
        return self.outputs
