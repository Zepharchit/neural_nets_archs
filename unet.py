import torch
import torch.nn as nn
import torch.nn.functional as F 

class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3,
                               padding=0,stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,
                               padding=0,stride=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    
    def forward(self,x):
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Upsample(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.convT = nn.ConvTranspose2d(in_dim,out_dim,kernel_size=2,stride=2)
    
    def forward(self,x):
        x = self.convT(x)
        return x

class Unet(nn.Module):
    def __init__(self,in_channel,filter_size,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.filter_size = filter_size
        self.out_channel = out_channel
        self.downsample = nn.ModuleList([DoubleConv(in_channel,filter_size),
                                         DoubleConv(filter_size,filter_size*2),
                                         DoubleConv(filter_size*2,filter_size*4),
                                         DoubleConv(filter_size*4,filter_size*8)])
        self.bridge = DoubleConv(filter_size*8,filter_size*16)
        self.residuals = list()
        self.mp = nn.MaxPool2d(2,2)
        self.upsamples = nn.ModuleList([
            Upsample(filter_size*16,filter_size*8),
            Upsample(filter_size*8,filter_size*4),
            Upsample(filter_size*4,filter_size*2),
            Upsample(filter_size*2,filter_size),
        ])
        self.decoder = nn.ModuleList([
           DoubleConv(filter_size*16,filter_size*8),
            DoubleConv(filter_size*8,filter_size*4),
            DoubleConv(filter_size*4,filter_size*2),
            DoubleConv(filter_size*2,filter_size),
       ])
        self.final = nn.Conv2d(filter_size,2,kernel_size=1,stride=1)
    def forward(self,x):
        #Down Conv
        for down in self.downsample:
            x = down(x)
            self.residuals.append(x)
            x = self.mp(x)
       #bridge
        x = self.bridge(x)
        self.residuals = self.residuals[::-1]
        for idx in range(len(self.upsamples)):
            x = self.upsamples[idx](x)
            res = self.residuals[idx]
            if x.shape!=res.shape:
                res = F.interpolate(res,size=x.shape[2:])
            x = torch.concat([x,res],dim=1)
            x = self.decoder[idx](x)
        
        x = self.final(x)
        print(x.shape)
        return x
    
net = Unet(3,64,1)
print(net)
net.forward(torch.rand((1,3,1024,1024)))