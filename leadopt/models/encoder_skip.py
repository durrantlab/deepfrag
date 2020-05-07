
import torch
import torch.nn as nn
import torch.nn.functional as F



class FullSkipV1(nn.Module):

    def __init__(self, input_channels, output_channels, c1=32, c2=64, c3=128, m=256):
        super(FullSkipV1, self).__init__()

        c_enc = [c1, c2, c3]

        self.encode = nn.ModuleList()
        prev = input_channels
        for i in range(3):
            block = nn.Sequential(
                nn.Conv3d(prev, c_enc[i], (3,3,3), padding=1),
                nn.ReLU(),
                nn.Conv3d(c_enc[i], c_enc[i], (3,3,3), padding=1),
                nn.ReLU(),
            )
            self.encode.append(block)
            prev = c_enc[i]

        self.encode_fc = nn.Sequential(
            nn.Conv3d(c_enc[2], m, (3,3,3), padding=1),
            nn.ReLU(),
        )

        self.decode_fc = nn.Sequential(
            nn.Conv3d(m, c_enc[2], (3,3,3), padding=1),
            nn.ReLU(),
        )

        self.decode = nn.ModuleList()
        for i in range(2,-1,-1):
            block = nn.Sequential(
                nn.Conv3d(c_enc[i] + (c_enc[i+1] if i != 2 else c_enc[i]), c_enc[i], (3,3,3), padding=1),
                nn.ReLU(),
                nn.Conv3d(c_enc[i], c_enc[i] if i != 0 else output_channels, (3,3,3), padding=1),
                nn.ReLU(),
            )
            self.decode.append(block)

        
    def forward(self, x):
        
        skip = []

        for i in range(3):
            x = self.encode[i](x)

            skip.append(x)
            x = nn.MaxPool3d((2,2,2))(x)

        x = self.encode_fc(x)

        x = self.decode_fc(x)

        for i in range(3):
            x = nn.Upsample(scale_factor=(2,2,2))(x)

            x = torch.cat([x, skip[2-i]], axis=1)

            x = self.decode[i](x)

        return x
    