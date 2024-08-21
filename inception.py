import torch
import torch.nn as nn

class IncepBlock(nn.Module):
    def __init__(self,inchannels,in1x1,red3x3,in3x3,red5x5,in5x5,pl):
        super(IncepBlock, self).__init__()
        self.inchannels = inchannels
        self.con1 = nn.Conv2d(inchannels,in1x1,kernel_size=1,stride=1)
        self.con2 = nn.Conv2d(inchannels,red3x3,kernel_size=1,stride=1)
        self.con3 = nn.Conv2d(inchannels,red5x5,kernel_size=1,stride=1)
        self.mp = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.conv3x3 = nn.Conv2d(red3x3,in3x3,kernel_size=3,stride=1,padding=1)
        self.conv5x5 = nn.Conv2d(red5x5,in5x5,kernel_size=5,stride=1,padding=2)
        self.con4 = nn.Conv2d(inchannels,pl,kernel_size=1,stride=1)
        self.rel = nn.ReLU()
        
    def forward(self,x):
        x1 = self.rel(self.con1(x))
        x2 = self.rel(self.conv3x3(self.rel(self.con2(x))))
        x3 = self.rel(self.conv5x5(self.rel(self.con3(x))))
        x4 = self.rel(self.con4(self.mp(x)))

        return torch.concat([x1, x2, x3, x4],axis=1)

class AuxOut(nn.Module):
    def __init__(self,in_ch,out_d):
        super(AuxOut, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.rel = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch,128,kernel_size=1,stride=1)
        self.drp = nn.Dropout(p=0.7)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,out_d)

    def forward(self,x):
        x = self.avg_pool(x)
        x = self.rel(self.conv1(x))
        x = x.view(x.size(0),-1)
        x = self.drp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x 
       
class Inception(nn.Module):
    def __init__(self,inchannel,output_dim,aux=None,training=None):
        super(Inception, self).__init__()
        self.inchannel = inchannel

        self.aux = aux
        self.training = training
        self.conv7x7 = nn.Conv2d(in_channels=inchannel,out_channels=64,stride=2,kernel_size=7,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv3x3 = nn.Conv2d(in_channels=64,out_channels=192,stride=1,kernel_size=3,padding=1)
        self.blk3a = IncepBlock(192,64,96,128,16,32,32)
        self.blk3b = IncepBlock(256,128,128,192,32,96,64)
        self.blk4a = IncepBlock(480,192,96,208,16,48,64)
        self.blk4b = IncepBlock(512,160,112,224,24,64,64)
        self.blk4c = IncepBlock(512,128,128,256,24,64,64)
        self.blk4d = IncepBlock(512,112,144,288,32,64,64)
        self.blk4e = IncepBlock(528,256,160,320,32,128,128)
        self.blk5a = IncepBlock(832,256,160,320,32,128,128)
        self.blk5b = IncepBlock(832,384,192,384,48,128,128)
        self.avg = nn.AvgPool2d(kernel_size=7,stride=1)
        self.drop = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024,out_features=output_dim)
        self.relu = nn.ReLU()

        if self.training and self.aux:
            self.a1 = AuxOut(512,output_dim)
            self.a2 = AuxOut(528,output_dim)

        else:
            self.a1 = None
            self.a2 = None

    def forward(self,x):
        #print(IncepBlock)
        x = self.relu(self.conv7x7(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3x3(x))
        x = self.maxpool(x)
        x = self.blk3a(x)
        x = self.blk3b(x)
        x = self.maxpool(x)
        x = self.blk4a(x)
        if self.training and self.aux:
            aux1 = self.a1(x)
        x = self.blk4b(x)
        x = self.blk4c(x)
        x = self.blk4d(x)
        if self.training and self.aux:
            aux2 = self.a2(x)
        x = self.blk4e(x)
        x = self.maxpool(x)
        x = self.blk5a(x)
        x = self.blk5b(x)
        x = self.drop(self.avg(x))
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        print(x.shape)
        return x,aux1,aux2
   
net = Inception(3,1000,aux=True,training=True)
net(torch.rand(1,3,224,224))