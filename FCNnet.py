import torch.nn as nn
import torch
from torchvision.models import vgg19_bn

#定义卷积块
class ConvBnLeakyBlock(nn.Module):
    def __init__(self,in_chanel,out_channel,kernel_size=3,padding=1,stride=1,padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chanel,out_channel,kernel_size=kernel_size,padding=padding,stride=stride,bias=False,padding_mode=padding_mode),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1,inplace=True)
        )
    def forward(self,input):
        return self.conv(input)
class FCNnet(nn.Module):
    def __init__(self,C=2,is_pretrained=True):
        super().__init__()
        backbone = vgg19_bn(is_pretrained).features
        self.down_1 = backbone[:27]#下采样8倍
        self.down_2 = backbone[27:40]#下采样16倍
        self.down_3 = backbone[40:]#下采样32
        self.conv_3 =nn.Sequential(ConvBnLeakyBlock(512,1024,kernel_size=1,padding=0,padding_mode="reflect"),
                                   ConvBnLeakyBlock(1024,512,kernel_size=1,padding=0,padding_mode="reflect"))
        self.up_3 = nn.ConvTranspose2d(512,512,kernel_size=4,padding=1,stride=2)
        self.conv_2=ConvBnLeakyBlock(512,512,kernel_size=1,padding=0,padding_mode="reflect")
        self.up_2 = nn.ConvTranspose2d(512,256,kernel_size=4,padding=1,stride=2)
        self.conv_1 = ConvBnLeakyBlock(256,256,kernel_size=1,padding=0,padding_mode="reflect")
        self.up_1 = nn.ConvTranspose2d(256,C,kernel_size=8,padding=0,stride=8)
    def forward(self,input):
        out_1 = self.down_1(input)
        out_2 = self.down_2(out_1)
        out_3 = self.down_3(out_2)
        out_3 = self.conv_3(out_3)
        out_3 = self.up_3(out_3)

        out_2 = self.conv_2(out_2)
        out = out_2+out_3
        out = self.up_2(out)

        out_1 = self.conv_1(out_1)
        out = out+out_1
        out = self.up_1(out)

        return out

if __name__=="__main__":
    net = FCNnet()
    input = torch.rand(2,3,416,416)
    predict = net(input)
    print(predict.size())



