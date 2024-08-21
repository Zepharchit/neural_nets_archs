import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channel,strides=1,id_block = None) -> None:
        super(ResBlock,self).__init__()
        self.upsample = 4
        self.strides = strides
        self.in_channels = 64
        self.out_channel = out_channel
        self.id_block = id_block
        self.conv1 = nn.Conv2d(in_channels,out_channel,kernel_size=1,stride=strides)
        self.conv3 = nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1,stride=1)
        self.conv1_2 = nn.Conv2d(out_channel,out_channel * self.upsample,kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel*self.upsample)
        self.relu = nn.ReLU()


        
    def forward(self,x):
        x_inp = x        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv3(x)))
        x = self.relu(self.bn3(self.conv1_2(x)))        
        if self.id_block is not None:
            x_inp = self.id_block(x_inp)
        x = x + x_inp
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self,img_channels,ResBlock,block_list,out_dim):
        super(ResNet,self).__init__()
        self.img_channels = img_channels
        self.in_channels = 64
        self.block_list = block_list
        self.out_dim = out_dim
        self.conv7 = nn.Conv2d(in_channels=img_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.mpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048,out_dim)
        self.block1 = self.get_blocks(ResBlock,middle_channels=64,i=block_list[0],strides=1)
        self.block2 = self.get_blocks(ResBlock,middle_channels=128,i=block_list[1],strides=2)
        self.block3 = self.get_blocks(ResBlock,middle_channels=256,i=block_list[2],strides=2)
        self.block4 = self.get_blocks(ResBlock,middle_channels=512,i=block_list[3],strides=2)
        self.rel = nn.ReLU()


    def get_blocks(self,ResBlock,middle_channels,i,strides):
        layers = []
        id_block = None
        if strides!=1 or self.in_channels!=middle_channels*4:
            id_block = nn.Sequential(nn.Conv2d(self.in_channels,middle_channels*4,kernel_size=1,stride=strides),
                                     nn.BatchNorm2d(middle_channels*4))
        layers.append(ResBlock(self.in_channels,middle_channels,strides,id_block))
        self.in_channels = middle_channels * 4
        #print(self.in_channels)
        for k in range(i-1):
            layers.append(ResBlock(self.in_channels,middle_channels,1,None))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.mpool(self.rel(self.bn(self.conv7(x))))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        print(x.shape)
        
        x = self.fc(x)
        return x

print(ResBlock)
net = ResNet(3,ResBlock,[3,4,6,3],100)
print(net(torch.rand(1,3,224,224)))    

'''
Here the ResBlock is being called/considered as a script
as all scripts have a __main__ method/function

'''