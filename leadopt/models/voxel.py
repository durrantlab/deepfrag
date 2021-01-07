# Copyright 2021 Jacob Durrant

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy
# of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F

Flatten = None
try:
    Flatten = nn.Flatten
except:
    from . import backport
    Flatten = backport.Flatten

class VoxelFingerprintNet(nn.Module):
    def __init__(self, in_channels, output_size, blocks=[32,64], fc=[2048], pad=True):
        super(VoxelFingerprintNet, self).__init__()

        blocks = list(blocks)
        fc = list(fc)

        self.blocks = nn.ModuleList()
        prev = in_channels
        for i in range(len(blocks)):
            b = blocks[i]
            parts = []
            parts += [
                nn.BatchNorm3d(prev),
                nn.Conv3d(prev, b, (3,3,3), padding=(1 if pad else 0)),
                nn.ReLU(),
                nn.Conv3d(b, b, (3,3,3), padding=(1 if pad else 0)),
                nn.ReLU(),
                nn.Conv3d(b, b, (3,3,3), padding=(1 if pad else 0)),
                nn.ReLU()
            ]
            if i != len(blocks)-1:
                parts += [
                    nn.MaxPool3d((2,2,2))
                ]

            self.blocks.append(nn.Sequential(*parts))
            prev = b

        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            Flatten(),
        )

        pred = []
        prev = blocks[-1]
        for f in fc + [output_size]:
            pred += [
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(prev, f)
            ]
            prev = f

        self.pred = nn.Sequential(*pred)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.reduce(x)
        x = self.pred(x)
        x = self.norm(x)

        return x
