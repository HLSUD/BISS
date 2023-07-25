"""
Basic DNN model for AAD
"""


import torch
import torch.nn.functional as F 
from torch import nn  # All neural network modules

def double_conv(in_channels, out_channels):
    conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    return conv_layers

def single_conv(in_channels, out_channels):
    conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    return conv_layers
### dnn with u-net shape
class u_net(nn.Module):
    def __init__(self, input_size, output_size=257, channels=16, bilinear=False):
        super(u_net, self).__init__()
        factor = 2 if bilinear else 1

        self.fc1 = nn.Linear(input_size,64*64*2) # -> transform to (256,256)

        # self.uflatten = nn.Unflatten(1, torch.Size([channels, 64, 64]))
        self.uflatten = nn.Unflatten(1, torch.Size([2, 64, 64]))
        self.single_conv1 = single_conv(2, channels)
        channels_base = 64
        ## down sampling
        self.double_conv1 = double_conv(channels, channels_base)
        self.pool = nn.MaxPool2d(2)
        self.double_conv2 = double_conv(channels_base, 2*channels_base)
        self.double_conv3 = double_conv(2*channels_base, 4*channels_base)
        self.double_conv4 = double_conv(4*channels_base, 8*channels_base)

        self.double_conv_mid = double_conv(8*channels_base, 16*channels_base//factor)

        # up sampling
        if bilinear:
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up1 = nn.ConvTranspose2d(16*channels_base, 8*channels_base, kernel_size=2, stride=2)
            self.up2 = nn.ConvTranspose2d(8*channels_base, 4*channels_base, kernel_size=2, stride=2)
            self.up3 = nn.ConvTranspose2d(4*channels_base, 2*channels_base, kernel_size=2, stride=2)
            self.up4 = nn.ConvTranspose2d(2*channels_base, channels_base, kernel_size=2, stride=2)

        self.double_conv5 = double_conv(16*channels_base, 8*channels_base // factor)
        self.double_conv6 = double_conv(8*channels_base, 4*channels_base // factor)
        self.double_conv7 = double_conv(4*channels_base, 2*channels_base // factor)
        self.double_conv8 = double_conv(2*channels_base, channels_base // factor)

        self.single_conv2 = single_conv(2*channels, 2)
        self.fc2 = nn.Linear(2 * 64 * 64, output_size)
        # self.fc2 = nn.Linear(channels_base * 64 * 64, 64 * 64 * 4)
        # self.fc3 = nn.Linear(64 * 64 * 4, output_size)
        self.dpt1 = nn.Dropout(p=0.5)
        self.dpt2 = nn.Dropout(p=0.3)

        self.initialize_weight()


    def initialize_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias,0)
            if isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.bias,0)
                nn.init.constant_(layer.weight,1)
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias,0)
                nn.init.kaiming_uniform_(layer.weight)


    
    def up_forward(self, x1, x2):

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = F.normalize(x,dim=1)

        x = self.fc1(x)
        # fc dropout
        # x = self.dpt1(x) ## eval 0

        # reshape
        x = self.uflatten(x)
        # x = x.reshape(x.shape[0], -1)
        # u-net
        x1 = self.double_conv1(x)
        x2 = self.pool(x1)
        x2 = self.double_conv2(x2)
        x3 = self.pool(x2)
        x3 = self.double_conv3(x3)
        x4 = self.pool(x3)
        x4 = self.double_conv4(x4)

        x5 = self.pool(x4)
        x5 = self.double_conv_mid(x5)

        x = self.up_forward(self.up1(x5),x4) ## check
        x = self.double_conv5(x)
        x = self.up_forward(self.up2(x),x3) ## check
        x = self.double_conv6(x)
        x = self.up_forward(self.up3(x),x2) ## check
        x = self.double_conv7(x)
        x = self.up_forward(self.up4(x),x1)
        x = self.double_conv8(x)

        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.dpt2(x) ## eval 0

        return x

if __name__ == '__main__':
    from torchinfo import summary
    model = u_net(input_size=32000, output_size=34480, channels=8).to("cuda")
    summary(model, input_size=(16,32000))