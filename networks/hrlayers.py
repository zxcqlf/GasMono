


from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *

class FFModule(nn.Module):
    def __init__(self, channels, scales, num_branch):
        super(FFModule).__init__()
        self.scale = scales
        self.out_ch = channels[int(scales)]
        self.in_ch = sum(channels)
        self.num_branches = num_branch

        self.dw_cnv = nn.Sequential(
            torch.nn.Conv2d(self.in_ch, self.out_ch, 1, 1, 0,  bias=False),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(self.out_ch, self.out_ch, 1, 1, 0, groups=self.out_ch, bias=False),
            torch.nn.Conv2d(self.out_ch, self.out_ch, 1, 1, 0,  bias=False),
            nn.SyncBatchNorm(self.in_ch, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.in_ch
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.SyncBatchNorm(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_inchannels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=num_inchannels[j],
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_inchannels[j], momentum=BN_MOMENTUM
                                    ),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    nn.SyncBatchNorm(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, inf):
        ff=[]
        for i in range(4):
            ff.append(F.interpolate(inf[self.scale], scale_factor=2**(int(i-self.scale)), mode="nearest"))
        aa = torch.cat(ff,1)
        output = self.dw_cnv(aa)
        return output

        


def upsample_disp(disp, mask, scale):
    """ Upsample flow field [H, W, 1] -> [H*2**scale, W*2**scale, 1] using convex combination """
    N, _, H, W = disp.shape
    mask = mask.view(N, 1, 9, 2**scale, 2**scale, H, W)
    mask = torch.softmax(mask, dim=2)
    PP = 3 - W%3

    up_flow = F.unfold(8 * disp, [3,3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, 1, H*2**scale, W*2**scale)



class AttModule(nn.Module):
    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(AttModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel
        if output_channel is not None:
            out_channel = output_channel
        reduction = 16
        channel = in_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = [high_features,low_features]
        #features += low_features
        features = torch.cat(features, 1)

        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = self.sigmoid(y)
        features = features * y.expand_as(features)

        return self.relu(self.conv_se(features))
