
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentEncoder(nn.Module):
    
    def __init__(self, input_channels, use_sigmoid=False, c1=32, c2=64, c3=128, h=2048, z=1024):
        super(LatentEncoder, self).__init__()

        self.use_sigmoid = use_sigmoid

        self.conv = nn.Sequential(
            nn.Conv3d(input_channels, c1, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c1, c1, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c1, c1, (3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(c1, c2, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c2, c2, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c2, c2, (3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(c2, c3, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c3, c3, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c3, c3, (3,3,3), padding=1),
            nn.ReLU(),
        )
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(c3 * 6 * 6 * 6, h),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h, z)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flat(x)
        x = self.fc(x)

        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x


class LatentDecoder(nn.Module):
    
    def __init__(self, output_channels, use_sigmoid=False, c1=32, c2=64, c3=128, h=2048, z=1024):
        super(LatentDecoder, self).__init__()

        self.use_sigmoid = use_sigmoid
        self.c3 = c3

        self.fc = nn.Sequential(
            nn.Linear(z, h),
            nn.ReLU(),
            nn.Linear(h, c3 * 6 * 6 * 6),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv3d(c3, c3, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c3, c3, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c3, c3, (3,3,3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(c3, c2, (3,3,3), stride=2, padding=1, output_padding=1),
            nn.Conv3d(c2, c2, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c2, c2, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c2, c2, (3,3,3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(c2, c1, (3,3,3), stride=2, padding=1, output_padding=1),
            nn.Conv3d(c1, c1, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c1, c1, (3,3,3), padding=1),
            nn.ReLU(),
            nn.Conv3d(c1, output_channels, (3,3,3), padding=1),
        )


    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.c3, 6, 6, 6)
        x = self.conv(x)

        if self.use_sigmoid:
            x = torch.sigmoid(x)

        return x
