# Define the UNet Model from scratch using plain pytorch
import torch
import torch.nn as nn


class NNUNet(nn.Module):
    def __init__(self, args):
        super(NNUNet, self).__init__()
        self.args = args
        # downsampling part
        self.DownConv1 = self.ContractBlock(self.args.in_channels, 32, 3, 1)
        self.DownConv2 = self.ContractBlock(32, 64, 3, 1)
        self.DownConv3 = self.ContractBlock(64, 128, 3, 1)
        self.DownConv4 = self.ContractBlock(128, 256, 3, 1)
        self.DownConv5 = self.ContractBlock(256, 512, 3, 1)
            
        # upsampling part
        self.UpConv5 = self.ExpandBlock(512, 256, 3, 1)
        self.UpConv4 = self.ExpandBlock(256*2, 128, 3, 1)
        self.UpConv3 = self.ExpandBlock(128*2, 64, 3, 1)
        self.UpConv2 = self.ExpandBlock(64*2, 32, 3, 1)
        self.UpConv1 = self.ExpandBlock(32*2, self.args.out_channels, 3, 1)
        
        # initialize weights
        self._initialize_weights()
        
    def forward(self, x):
         
        DownConv1 = self.DownConv1(x)
        DownConv2 = self.DownConv2(DownConv1) 
        DownConv3 = self.DownConv3(DownConv2) 
        DownConv4 = self.DownConv4(DownConv3) 
        DownConv5 = self.DownConv5(DownConv4)
        UpConv5   = self.UpConv5 (DownConv5)
        UpConv4   = self.UpConv4 (torch.cat([UpConv5, DownConv4], 1))
        UpConv3   = self.UpConv3 (torch.cat([UpConv4, DownConv3], 1))
        UpConv2   = self.UpConv2 (torch.cat([UpConv3, DownConv2], 1))
        UpConv1   = self.UpConv1 (torch.cat([UpConv2, DownConv1], 1))
        
        return UpConv1
        
        
    def ContractBlock(self, in_channels, out_channels, kernel_size, padding):
        
        contract = nn.Sequential(            
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
        return contract



    def ExpandBlock(self, in_channels, out_channels, kernel_size, padding):
        
        expand = nn.Sequential(
            
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      stride=1, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1) )

        return expand
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()