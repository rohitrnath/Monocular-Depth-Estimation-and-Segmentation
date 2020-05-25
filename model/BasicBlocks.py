import torch
import torch.nn as nn
import torch.nn.functional as F

def PointwiseConv(in_panels, out_panels, last=False):
    if last:
      return ( Sequential(nn.Conv2d( in_panels, out_panels, kernel_size=(1,1))) )
    else:
      return(nn.Sequential(
              nn.Conv2d( in_panels, out_panels, kernel_size=(1,1)),
              nn.ReLU(),
              nn.BatchNorm2d(out_panels)
            ))
def DepthwiseConv(in_panels, out_panels, dilation=1):
    DepthConvBlock = nn.Sequential(
            nn.Conv2d( in_panels, in_panels, kernel_size=(3,3), padding=dilation, groups=in_panels,  dilation = dilation),
            PointwiseConv( in_panels, out_panels),
        )
    return DepthConvBlock

def TransConv(in_panels, out_panels, dilation=1):
    #o = (i -1)*s - 2*p + k + output_padding 
    TransConvBlock = nn.Sequential(
            nn.ConvTranspose2d( in_panels, in_panels, kernel_size=(3,3), padding=dilation, output_padding=1, stride=2, groups=in_panels,  dilation = dilation),
            PointwiseConv( in_panels, out_panels),
        )
    return TransConvBlock

def InitConv(inp, oup):
    return (nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(),

        # nn.Conv2d(oup, oup, 4, 4, 114, bias=False),
        nn.Conv2d(oup, oup, 3, 2, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=4, stride=4, padding=1)
    ))