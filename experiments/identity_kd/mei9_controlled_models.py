"""Torch-only standard ResNet18 and a statistics MLP for the controlled study.

ResNet18 uses the conventional [2,2,2,2] BasicBlock graph, allowing strict
torchvision state-dict/forward parity verification without local torchvision.
"""
import torch
from torch import nn
from torch.nn import functional as F
def conv(a,b,stride=1):return nn.Conv2d(a,b,3,stride,1,bias=False)
class Block(nn.Module):
    def __init__(self,a,b,stride=1):
        super().__init__();self.conv1=conv(a,b,stride);self.bn1=nn.BatchNorm2d(b);self.relu=nn.ReLU(inplace=True);self.conv2=conv(b,b);self.bn2=nn.BatchNorm2d(b);self.downsample=nn.Sequential(nn.Conv2d(a,b,1,stride,bias=False),nn.BatchNorm2d(b)) if stride!=1 or a!=b else None
    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x)));y=self.bn2(self.conv2(y));return self.relu(y+(x if self.downsample is None else self.downsample(x)))
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(3,64,7,2,3,bias=False);self.bn1=nn.BatchNorm2d(64);self.relu=nn.ReLU(inplace=True);self.maxpool=nn.MaxPool2d(3,2,1);self.layer1=nn.Sequential(Block(64,64),Block(64,64));self.layer2=nn.Sequential(Block(64,128,2),Block(128,128));self.layer3=nn.Sequential(Block(128,256,2),Block(256,256));self.layer4=nn.Sequential(Block(256,512,2),Block(512,512));self.avgpool=nn.AdaptiveAvgPool2d((1,1));self.fc=nn.Linear(512,192)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))));x=self.layer4(self.layer3(self.layer2(self.layer1(x))));return F.normalize(self.fc(self.avgpool(x).flatten(1)),dim=1)
class StatsMLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(720,256),nn.GELU(),nn.Linear(256,192))
    def forward(self,x):return F.normalize(self.net(x),dim=1)
