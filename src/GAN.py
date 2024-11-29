import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU 20%
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Convolution
            nn.BatchNorm2d(out_channels)  # Normalisation 
        )

    def forward(self, x):
        return self.conv(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return self.bottleneck(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )


    def forward(self, x):
        return self.decoder(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = ConvBlock(3, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 512)
        self.encoder6 = ConvBlock(512, 512)
        self.encoder7 = ConvBlock(512, 512)

        self.bottleneck = Bottleneck(512)
        
        self.decoder7 = DecoderBlock(512, 512)
        self.decoder6 = DecoderBlock(512, 512)
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 3)  

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        x6 = self.encoder6(x5)
        x7 = self.encoder7(x6)

        # Bottleneck
        bottleneck_output = self.bottleneck(x7)

        # Decoder
        d7 = self.decoder7(bottleneck_output)
        d6 = self.decoder6(d7 + x6) 
        d5 = self.decoder5(d6 + x5) 
        d4 = self.decoder4(d5 + x4)  
        d3 = self.decoder3(d4 + x3)  
        d2 = self.decoder2(d3 + x2)  
        output = self.decoder1(d2 + x1)  

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),  # 1024 -> 512
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 512 -> 256
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),  # 128 -> 127
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # 127 -> 126
        )

    def forward(self, x):
        return self.model(x)