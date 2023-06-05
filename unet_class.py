import torch
import torch.nn as nn
import torch.nn.functional as F


# (701, 1024, 1024, 1)
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()

        # Contracting path
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(512)
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm3d(1024)

        # Expanding path
        self.upconv6 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm3d(512)
        self.upconv7 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm3d(256)
        self.upconv8 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm3d(128)
        self.upconv9 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm3d(64)

        # Output layer
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        conv1 = F.relu(self.bn1(self.conv1(x)))
        conv2 = F.relu(self.bn2(self.conv2(F.max_pool3d(conv1, kernel_size=2, stride=2))))
        conv3 = F.relu(self.bn3(self.conv3(F.max_pool3d(conv2, kernel_size=2, stride=2))))
        conv4 = F.relu(self.bn4(self.conv4(F.max_pool3d(conv3, kernel_size=2, stride=2))))
        conv5 = F.relu(self.bn5(self.conv5(F.max_pool3d(conv4, kernel_size=2, stride=2))))

        # Expanding path
        upconv6 = self.upconv6(conv5)
        conv6 = F.relu(self.bn6(self.conv6(torch.cat([upconv6, conv4], dim=1))))
        upconv7 = self.upconv7(conv6)
        conv7 = F.relu(self.bn7(self.conv7(torch.cat([upconv7, conv3], dim=1))))
        upconv8 = self.upconv8(conv7)
        conv8 = F.relu(self.bn8(self.conv8(torch.cat([upconv8, conv2], dim=1))))
        upconv9 = self.upconv9(conv8)
        conv9 = F.relu(self.bn9(self.conv9(torch.cat([upconv9, conv1], dim=1))))

        # Output layer
        output = self.output(conv9)

        return output
        # # Encoder
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=2, stride=2),
        #     nn.Conv3d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     #nn.MaxPool3d(kernel_size=2, stride=2)
        # )
        # # Middle
        # self.middle = nn.Sequential(
        #     nn.Conv3d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2),
        # )
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Conv3d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(256, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(128),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
        #     nn.Conv3d(64, out_channels, kernel_size=1)
        # )
        # # Skip connections
        # self.skip_connection = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv3d(256, 128, kernel_size=1),
        #         nn.BatchNorm3d(128),
        #         nn.ReLU(inplace=True),
        #         nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
        #     ),
        #     nn.Sequential(
        #         nn.Conv3d(128, 64, kernel_size=1),
        #         nn.BatchNorm3d(64),
        #         nn.ReLU(inplace=True),
        #         nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
        #     )
        # ])

    # def forward(self, x):
    #     # Encoder
    #     x1 = self.encoder(x)
    #     # Middle
    #     x2 = self.middle(x1)
    #     # Decoder with skip connections
    #     x3 = self.decoder(x2 + self.skip_connection[1](x2))
    #     x3 = self.skip_connection[0](x3 + x1)
    #     return x3

    # def forward(self, x):
    #     # Encoder
    #     x1 = self.encoder(x)
    #     print(f'x1={x1.shape}')
    #     # Middle
    #     x2 = self.middle(x1)
    #     print(f'x2={x2.shape}')
    #     # Decoder with skip connections
    #     x3 = self.decoder(x2)
    #     print(f'x3={x3.shape}')
    #     x3 = torch.cat([x3, self.skip_connection[0](x2)], dim=1)
    #     print(f'x3cc={x3.shape}')
    #     x3 = self.skip_connection[1](x3)
    #     print(f'x3after_skip1={x3.shape}')
    #     x3 = x3 + x1
    #     print(f'x3+x1={x3.shape}')
    #     return x3
