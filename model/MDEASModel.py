import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicBlocks import InitConv, DepthwiseConv, PointwiseConv, TransConv
# from Decoder import MaskDecoderBlock, DepthDecoderBlock


class DenseBlock(nn.Module):

    def __init__(self, in_panels, out_panels):
        super(DenseBlock, self).__init__()
        self.DepthConvBlock1 = DepthwiseConv(in_panels, in_panels)
        self.DepthConvBlock2 = DepthwiseConv(in_panels, in_panels)
        self.DepthConvBlock3 = DepthwiseConv(in_panels, in_panels)
        self.one             = PointwiseConv(in_panels, out_panels)

    def forward(self, x):

        x1  = self.DepthConvBlock1(x)
        x2  = self.DepthConvBlock2(x+x1)
        x3  = self.DepthConvBlock3(x+x1+x2)
        out = self.one(x+x1+x2+x3)

        return out


#Initial Block
class InitBlock(nn.Module):

    def __init__(self):
        super(InitBlock, self).__init__()
        self.InitBlock_1  = InitConv( 3, 32)
        self.InitBlock_2  = InitConv( 3, 32)

 
    def forward(self,  bg, fg_bg):

        InitOut_1 = self.InitBlock_1(bg)
        InitOut_2 = self.InitBlock_2(fg_bg)

        InitOut   = torch.cat([InitOut_1, InitOut_2], 1)

        return InitOut, InitOut_1, InitOut_2


# Encoder Block
'''class EncoderBlock(nn.Module):
  
  def __init__(self):
    super(EncoderBlock, self).__init__()

    self.DenseBlock_1 = DenseBlock( 64, 128)
    self.pool_1       = nn.MaxPool2d(2)

    self.DenseBlock_2 = DenseBlock( 128, 256)
    self.pool_2       = nn.MaxPool2d(2)

    self.DenseBlock_3 = DenseBlock( 256, 256)
    self.pool_3       = nn.MaxPool2d(2)


  def forward(self, x):

    EC1 = self.pool_1(self.DenseBlock_1(x))
    EC2 = self.pool_1(self.DenseBlock_2(EC1))
    EC3 = self.pool_1(self.DenseBlock_3(EC2))
    out = EC3
    return out, EC1, EC2'''


# BottleNeckBlock - Dilation Block 
'''
o = output
p = padding
k = kernel_size
s = stride
d = dilation.  o = [i + 2*p - k - (k-1)*(d-1)]/s + 1'''
class BottleNeckBlock(nn.Module):
  
  def __init__(self, in_panels=256, mid_panels=128, out_panels_1=128, out_panels_2=256):
    super(BottleNeckBlock, self).__init__()
    self.one             = PointwiseConv(in_panels, mid_panels)

    self.DilationBlock_1 = DepthwiseConv(mid_panels, mid_panels)
    self.DilationBlock_2 = DepthwiseConv(mid_panels, mid_panels, dilation=3)
    self.DilationBlock_3 = DepthwiseConv(mid_panels, mid_panels, dilation=6)
    self.DilationBlock_4 = DepthwiseConv(mid_panels, mid_panels, dilation=9)

    self.one1            = PointwiseConv(mid_panels*4, out_panels_1)
    self.one2            = PointwiseConv(mid_panels*4, out_panels_2)

  def forward(self, x):
    x0  = self.one(x)

    x1  = self.DilationBlock_1(x0)
    x2  = self.DilationBlock_2(x0)
    x3  = self.DilationBlock_3(x0)
    x4  = self.DilationBlock_4(x0)

    out = torch.cat((x1,x2,x3,x4), 1)

    outM = self.one1(out)
    outD = self.one2(out)

    return outM,outD

####################################UP-sampling Blocks Used############################################
### N X N convolution or interpolation (nearest interpolation giving more accuracy than bilinear interpolation)
class NNConv(nn.Module):

    def __init__(self, in_panels, out_panels):
        super(NNConv, self).__init__()
        self.DenseBlock     = DenseBlock( in_panels, out_panels)

    def forward(self, x):
        # x = F.pixel_shuffle(x, 2)
        out     =   self.DenseBlock(x)
        out     =   nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        return out

### Pixel-shufling to upsample
class ShuffleConv(nn.Module):

    def __init__(self, in_panels, out_panels):
        super(ShuffleConv, self).__init__()
        self.DenseBlock      = DenseBlock( in_panels, out_panels)
        self.pixel_shuffle   = nn.PixelShuffle(2)

    def forward(self, x):
        # x = F.pixel_shuffle(x, 2)
        out     =   self.pixel_shuffle(self.DenseBlock(x))
        return out

### De-Convolution or transpose convolution
class DeConv(nn.Module):

    def __init__(self, in_panels, out_panels):
        super(DeConv, self).__init__()
        self.DenseBlock = DenseBlock( in_panels, out_panels)
        self.DeConv     = nn.TransConv( out_panels, out_panels)

    def forward(self, x):
        # x = F.pixel_shuffle(x, 2)
        out     =   self.DeConv(self.DenseBlock(x))
        return out
####################################################################################################

### Depth Decoder Block
class DepthDecoderBlock(nn.Module):

    def __init__(self):
        super(DepthDecoderBlock, self).__init__()
        self.DecoderBlock_1  = NNConv( 256, 256)
        self.DecoderBlock_2  = NNConv( 256, 128)
        self.DecoderBlock_3  = ShuffleConv( 128, 128)
        self.DecoderBlock_4  = ShuffleConv( 32,  64)
        self.LastDepthBlock  = nn.Sequential(
                                nn.Conv2d( 16, 1, kernel_size=(1,1))
                              )

 
    def forward(self, In, EC1, EC2, InitOut_2):
        # x = F.pixel_shuffle(x, 2)
        DC1  = self.DecoderBlock_1(In)
        DC1 += EC2
        DC2  = self.DecoderBlock_2(DC1)
        DC2 += EC1
        DC3  = self.DecoderBlock_3(DC2)
        DC3 += InitOut_2
        DC4 = self.DecoderBlock_4(DC3)

        out= self.LastDepthBlock(DC4)

        return out

#### Mask-Decoder Block
class MaskDecoderBlock(nn.Module):

    def __init__(self):
        super(MaskDecoderBlock, self).__init__()
        self.DecoderBlock_1  = NNConv( 128, 128)
        self.DecoderBlock_2  = NNConv( 128, 128)
        self.DecoderBlock_3  = NNConv( 128, 64)
        self.DecoderBlock_4  = ShuffleConv( 64,  64)
        self.LastMaskBlock   = nn.Sequential(
                                nn.Conv2d( 16, 1, kernel_size=(1,1))
                              )

 
    def forward(self, In, EC1, InitOut):
        # x = F.pixel_shuffle(x, 2)
        DC1  = self.DecoderBlock_1(In)
        DC2  = self.DecoderBlock_2(DC1)
        DC2 += EC1
        DC3  = self.DecoderBlock_3(DC2)
        DC3 += InitOut
        DC4 = self.DecoderBlock_4(DC3)

        out= self.LastMaskBlock(DC4)

        out = torch.sigmoid(out)
        
        return out




#############################################MAIN MODEL##################################################
class MDEASModel(nn.Module):
    def __init__(self):
        super(MDEASModel, self).__init__()
#################### Initial Block #################
        self.InitBlock_1  = InitConv( 3, 32)
        self.InitBlock_2  = InitConv( 3, 32)

################## Encoder Blocks ##################       
        self.DenseBlock_1 = DenseBlock( 64, 128)
        self.pool_1       = nn.MaxPool2d(2)

        self.DenseBlock_2 = DenseBlock( 128, 256)
        self.pool_2       = nn.MaxPool2d(2)

        self.DenseBlock_3 = DenseBlock( 256, 256)
        self.pool_3       = nn.MaxPool2d(2)

################# Bottle-Neck Block #################
        self.BottleNeck   = BottleNeckBlock()

################## Depth Decoder ####################
        self.DepthDecoder = DepthDecoderBlock()

##################### Mask Decoder ##################
        self.MaskDecoder  = MaskDecoderBlock()


    def forward(self, bg, fg_bg):
### Initial Block
        InitOut_1 = self.InitBlock_1(bg)
        InitOut_2 = self.InitBlock_2(fg_bg)
        InitOut   = torch.cat([InitOut_1, InitOut_2], 1)

### Encoder Block
        EC1_out = self.pool_1(self.DenseBlock_1(InitOut))
        EC2_out = self.pool_1(self.DenseBlock_2(EC1_out))
        EC3_out = self.pool_1(self.DenseBlock_3(EC2_out2))

### Bottleneck Block --(Dilation block)
        MaskBranch, DepthBranch   = self.BottleNeck(EC3_out)
### Depth Decoder Block
        depth_out  = self.DepthDecoder( DepthBranch, EC1_out, EC2_out, InitOut_2)

### Mask Decoder Block
        mask_out   = self.MaskDecoder( MaskBranch, EC1_out, InitOut)

        return depth_out, mask_out

###############################################################################################################






#########################nitialize kernel weights with Gaussian distributions##################################
import math
def weights_init(m):

    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
################################################################################################################