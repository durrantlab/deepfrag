
import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelFingerprintNet(nn.Module):
    
    def __init__(self, in_channels, output_size, sigmoid=True, f1=32, f2=64, f3=128):
        super(VoxelFingerprintNet, self).__init__()

        self.sigmoid = sigmoid
        
        self.features = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.BatchNorm3d(f1),
            nn.Conv3d(f1, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.BatchNorm3d(f2),
            nn.Conv3d(f2, f3, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f3, f3, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f3, f3, (3,3,3)),
            nn.ReLU(),
        )
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
        )
        self.pred = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(f3, output_size),
        )
        self.norm = nn.Sigmoid()
    
    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = self.pred(x)

        if self.sigmoid:
            x = self.norm(x)
        
        return x

class VoxelFingerprintNet2(nn.Module):
    '''smaller model'''
    def __init__(self, in_channels, output_size, sigmoid=False, batchnorm=False, f1=32, f2=64):
        super(VoxelFingerprintNet2, self).__init__()

        self.sigmoid = sigmoid
        self.batchnorm = batchnorm

        features = []
        if batchnorm:
            features.append(nn.BatchNorm3d(in_channels))
        features += [
            nn.Conv3d(in_channels, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
        ]
        if batchnorm:
            features.append(nn.BatchNorm3d(f1))
        features += [
            nn.Conv3d(f1, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU()
        ]

        self.features = nn.Sequential(*features)
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
        )
        self.pred = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(f2, output_size),
        )
        self.norm = nn.Sigmoid()
    
    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = self.pred(x)

        if self.sigmoid:
            x = self.norm(x)
        
        return x

class VoxelFingerprintNet2b(nn.Module):
    '''smaller model'''
    def __init__(self, in_channels, output_size, blocks=[32,64], fc=[2048], pad=True):
        super(VoxelFingerprintNet2b, self).__init__()

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
            nn.Flatten(),
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

class VoxelFingerprintNet3(nn.Module):
    '''smaller model'''
    def __init__(self, in_channels, in_fingerprint, f1=32, f2=64, r1=256, r2=256, p1=256):
        super(VoxelFingerprintNet3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(in_channels, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(f1, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU(),
        )
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
        )
        self.process = nn.Sequential(
            nn.Linear(in_fingerprint, r1),
            nn.ReLU(),
            nn.Linear(r1, r2),
            nn.ReLU(),
        )
        self.pred = nn.Sequential(
            nn.Dropout(),
            nn.Linear(r2 + f2, p1),
            nn.ReLU(),
            nn.Linear(p1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, grid, fp):
        x = self.features(grid)
        x = self.reduce(x)

        y = self.process(fp)

        g = torch.cat([x,y], axis=1)

        o = self.pred(g)

        return o

class VoxelFingerprintNet4(nn.Module):
    '''smaller model'''
    def __init__(self, in_channels, output_size, sigmoid=False, batchnorm=False, f1=32, f2=64):
        super(VoxelFingerprintNet4, self).__init__()

        self.sigmoid = sigmoid
        self.batchnorm = batchnorm

        features = []
        if batchnorm:
            features.append(nn.BatchNorm3d(in_channels))
        features += [
            nn.Conv3d(in_channels, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f1, f1, (3,3,3)),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
        ]
        if batchnorm:
            features.append(nn.BatchNorm3d(f1))
        features += [
            nn.Conv3d(f1, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU(),
            nn.Conv3d(f2, f2, (3,3,3)),
            nn.ReLU()
        ]

        self.features = nn.Sequential(*features)
        self.reduce = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
        )
        self.pred = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(f2, output_size),
        )
        self.norm = nn.Sigmoid()
        self.attention = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(f2, output_size),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)

        fp = self.pred(x)
        if self.sigmoid:
            fp = self.norm(fp)

        att = self.attention(x)
        
        return fp, att
