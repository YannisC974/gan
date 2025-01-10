import torch
import torch.nn as nn

# Bloc de convolution utilisé dans l'encodeur
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Convolution
            nn.BatchNorm2d(out_channels),  # Normalisation
            nn.LeakyReLU(0.2, inplace=True)  # Activation LeakyReLU
        )

    def forward(self, x):
        return self.conv(x)


# Bloc Bottleneck avec convolution et déconvolution
class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),  # Réduction
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),  # Expansion
            nn.BatchNorm2d(in_channels),  # Normalisation
            nn.LeakyReLU(0.2, inplace=True)  # Activation
        )

    def forward(self, x):
        return self.bottleneck(x)


# Bloc de décodeur utilisé dans le décodeur UNet
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),  # Déconvolution
            nn.BatchNorm2d(out_channels),  # Normalisation
            nn.ReLU(inplace=True),  # Activation
            nn.Dropout(0.2)  # Régularisation
        )

    def forward(self, x):
        return self.decoder(x)


# Modèle UNet pour les tâches de segmentation
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encodeur
        self.encoder1 = ConvBlock(1, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 512)

        # Bottleneck
        self.bottleneck = Bottleneck(512)

        # Décodeur
        self.decoder5 = DecoderBlock(1024, 512)
        self.decoder4 = DecoderBlock(1024, 256)
        self.decoder3 = DecoderBlock(512, 128)
        self.decoder2 = DecoderBlock(256, 64)
        self.decoder1 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encodeur
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # Bottleneck
        bottleneck_output = self.bottleneck(x5)

        # Décodeur avec concaténation des connexions résiduelles
        d5 = self.decoder5(torch.cat([bottleneck_output, x5], dim=1))
        d4 = self.decoder4(torch.cat([d5, x4], dim=1))
        d3 = self.decoder3(torch.cat([d4, x3], dim=1))
        d2 = self.decoder2(torch.cat([d3, x2], dim=1))
        output = self.decoder1(d2)

        return output


# Modèle Discriminateur pour les tâches GAN
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)  # Sortie de la prédiction
        )

    def forward(self, x):
        return self.model(x)
