from torch import nn
#from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from collections import OrderedDict
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.model_zoo as model_zoo
from .pnasnet import PNASNet5Large
import pretrainedmodels
import torch.nn.functional as F

#from modules.wider_resnet import WiderResNet
from .resnext import *
"""
This script has been taken (and modified) from :
https://github.com/ternaus/TernausNet

@ARTICLE{arXiv:1801.05746,
         author = {V. Iglovikov and A. Shvets},
          title = {TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation},
        journal = {ArXiv e-prints},
         eprint = {1801.05746}, 
           year = 2018
        }
"""

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class NoOperation(nn.Module):
    def forward(self, x):
        return x


class DecoderBlock_old(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
	    nn.BatchNorm2d(out_channels), ##me added
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = ConvBn2d(in_channels, middle_channels)
        self.conv2 = ConvBn2d(middle_channels, out_channels)
        #self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        if e is not None:
            x = torch.cat([x,e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = x*g1 + x*g2
        return x




class UNet11(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


def unet11(pretrained=False, **kwargs):
    """
    pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
            carvana - all weights are pre-trained on
                Kaggle: Carvana dataset https://www.kaggle.com/c/carvana-image-masking-challenge
    """
    model = UNet11(pretrained=pretrained, **kwargs)

    if pretrained == 'carvana':
        state = torch.load('TernausNet.pt')
        model.load_state_dict(state['model'])
    return model


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                #nn.BatchNorm2d(out_channels), ##me added
		        nn.ReLU(inplace=True)
                
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class DecoderCenter(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderCenter, self).__init__()
        self.in_channels = in_channels

        
        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
		nn.BatchNorm2d(out_channels), ##me added
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
		#nn.BatchNorm2d(out_channels), ##me added
                #nn.ReLU(inplace=True) ##me added
 
            )

    def forward(self, x):
        return self.block(x)

class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(dec0)


class UNetVGG16(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, dropout_2d=0.2, pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder[0],
                                   self.relu,
                                   self.encoder[2],
                                   self.relu)

        self.conv2 = nn.Sequential(self.encoder[5],
                                   self.relu,
                                   self.encoder[7],
                                   self.relu)

        self.conv3 = nn.Sequential(self.encoder[10],
                                   self.relu,
                                   self.encoder[12],
                                   self.relu,
                                   self.encoder[14],
                                   self.relu)

        self.conv4 = nn.Sequential(self.encoder[17],
                                   self.relu,
                                   self.encoder[19],
                                   self.relu,
                                   self.encoder[21],
                                   self.relu)

        self.conv5 = nn.Sequential(self.encoder[24],
                                   self.relu,
                                   self.encoder[26],
                                   self.relu,
                                   self.encoder[28],
                                   self.relu)

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        return self.final(F.dropout2d(dec1, p=self.dropout_2d))

class UNetResNet(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
         
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 *2, num_filters * 8, False)
        
        self.dec5 =  DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8,   is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        #pool = self.pool(conv5) # deleted pooling
        #center = self.center(pool)
        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))

        
class UNetResNet_wo_pool(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        #print('input_adjust ', input_adjust.shape, '\ncenter ' , center.shape, '\ndec1: ', dec1.shape)
        return self.final(dec1)

class UNetResNext_wo_pool(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = pretrainedmodels.se_resnext50_32x4d() #torchvision.models.resnet152(pretrained=pretrained)

        self.pool = nn.MaxPool2d(2, 2)
        bottom_channel_nr = 512*4

        self.input_adjust = nn.Sequential(self.encoder.layer0.conv1,
                                          self.encoder.layer0.bn1,
                                          self.encoder.layer0.relu1)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        print('input_adjust ', input_adjust.shape, '\ncenter ' , center.shape, '\ndec1: ', dec1.shape, self.final(dec1).shape)
        return self.final(dec1)

class UNetResNetAttentionv2(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=True):
        super(UNetResNetAttention, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.resnet = pretrainedmodels.se_resnext50_32x4d()
        bottom_channel_nr = 2048
        self.encoder1 = EncoderBlock(
            nn.Sequential(self.resnet.layer0.conv1, self.resnet.layer0.bn1, self.resnet.layer0.relu1),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)

        center_block = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, bottom_channel_nr//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, bottom_channel_nr//2)

        self.decoder5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,  num_filters * 16, 64)
        self.decoder4 = DecoderBlock(64 + bottom_channel_nr // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlock(64 + bottom_channel_nr // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlock(64 + bottom_channel_nr // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlock(64, num_filters, 64)

        self.final = nn.Conv2d(64, 2, kernel_size=1)
        
        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) ; #print('d1:', d1.size())

        
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear'),
        ],   dim=1) 
        

        #f = F.dropout2d(f, p=self.dropout_2d)
        #print (self.logit(d1).shape)
        return self.logit(f)


class UNetResNetAttention(nn.Module):
    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=True):
        super(UNetResNetAttention, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d


        self.pool = nn.MaxPool2d(2, 2)

        self.resnet = pretrainedmodels.se_resnext50_32x4d()
        bottom_channel_nr = 2048
        
        conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        conv1.weight = self.resnet.layer0.conv1.weight
        '''
        self.encoder1 = nn.Sequential(conv1,
                                          self.resnet.layer0.bn1,
                                          self.resnet.layer0.relu1
                                          ,self.pool
                                          )
        '''
        self.encoder1 = EncoderBlock(
            nn.Sequential(conv1, self.resnet.layer0.bn1, self.resnet.layer0.relu1,self.pool),
            num_filters*2
        )
        self.encoder2 = EncoderBlock(self.resnet.layer1, bottom_channel_nr//8)
        self.encoder3 = EncoderBlock(self.resnet.layer2, bottom_channel_nr//4)
        self.encoder4 = EncoderBlock(self.resnet.layer3, bottom_channel_nr//2)
        self.encoder5 = EncoderBlock(self.resnet.layer4, bottom_channel_nr)

        center_block = nn.Sequential(
            ConvBn2d(bottom_channel_nr, bottom_channel_nr, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(bottom_channel_nr, bottom_channel_nr//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.center = EncoderBlock(center_block, bottom_channel_nr//2)

        self.decoder5 = DecoderBlock(bottom_channel_nr + bottom_channel_nr // 2,  num_filters * 16, 64)
        self.decoder4 = DecoderBlock(64 + bottom_channel_nr // 2,  num_filters * 8,  64)
        self.decoder3 = DecoderBlock(64 + bottom_channel_nr // 4,  num_filters * 4,  64)
        self.decoder2 = DecoderBlock(64 + bottom_channel_nr // 8, num_filters * 2,  64)
        self.decoder1 = DecoderBlock(64, num_filters, 64)

        self.final = nn.Conv2d(64, 2, kernel_size=1)
        
        self.logit = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.encoder1(x) #; print('x:', x.size())
        e2 = self.encoder2(x) #; print('e2:', e2.size())
        e3 = self.encoder3(e2) #; print('e3:', e3.size())
        e4 = self.encoder4(e3) #; print('e4:', e4.size())
        e5 = self.encoder5(e4) #; print('e5:', e5.size())

        center = self.center(e5) #; print('center:', center.size())

        d5 = self.decoder5(center, e5) #; print('d5:', d5.size())
        d4 = self.decoder4(d5, e4) #; print('d4:', d4.size())
        d3 = self.decoder3(d4, e3) #; print('d3:', d3.size())
        d2 = self.decoder2(d3, e2) #; print('d2:', d2.size())
        d1 = self.decoder1(d2) ; #print('d1:', d1.size())

        '''
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear'),
            F.upsample(d3, scale_factor=4, mode='bilinear'),
            F.upsample(d4, scale_factor=8, mode='bilinear'),
            F.upsample(d5, scale_factor=16, mode='bilinear'),
        ],   dim=1) 
        '''

        #f = F.dropout2d(f, p=self.dropout_2d)
        #print (self.logit(d1).shape)
        return self.final(d1)
class EncoderBlock(nn.Module):
    def __init__(self, block, out_channels):
        super(EncoderBlock, self).__init__()
        self.block = block
        self.out_channels = out_channels
        self.spatial_gate = SpatialAttentionGate(out_channels)
        self.channel_gate = ChannelAttentionGate(out_channels)

    def forward(self, x):
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)

        return x*g1 + x*g2
class ChannelAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SpatialAttentionGate(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SpatialAttentionGate, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, 1, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        #print(x.size())
        return x

class UNetResNext_wo_pool_hyper(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = pretrainedmodels.se_resnext50_32x4d() #torchvision.models.resnet152(pretrained=pretrained)

        self.pool = nn.MaxPool2d(2, 2)
        bottom_channel_nr = 512*4

        self.input_adjust = nn.Sequential(self.encoder.layer0.conv1,
                                          self.encoder.layer0.bn1,
                                          self.encoder.layer0.relu1)

        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.dec4 = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)

        
        self.final = nn.Conv2d(num_filters * 2 * 2, num_classes, kernel_size=1)
        self._mask_out = nn.Sequential(nn.Conv2d(704, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        center = self.conv4(conv3)

        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        hcol = torch.cat([dec1, F.upsample(dec2, scale_factor=2, mode='bilinear'),         #,align_corners=False
                                F.upsample(dec3, scale_factor=4, mode='bilinear'),         #,align_corners=False
                                F.upsample(dec4, scale_factor=8, mode='bilinear')], dim=1) #,align_corners=False
        #hcol = F.dropout2d(hcol, p = 0.5)
        #print('input_adjust ', input_adjust.shape, '\ncenter ' , center.shape, '\ndec1: ', dec1.shape)
        #print('hcol ', hcol.shape,   '\nout ', self._mask_out(hcol).shape)
        return self._mask_out(hcol)


class UNetResNext50(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        self.encoder = pretrainedmodels.se_resnext50_32x4d() #torchvision.models.resnet152(pretrained=pretrained)
        bottom_channel_nr = 512*4
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        #self.input_adjust =  nn.Sequential(self.encoder.layer0, self.pool)
     
        self.input_adjust =  self.encoder.layer0
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 *2, num_filters * 8, False)
        
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8,   is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        conv1 = self.conv1(input_adjust)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        center = self.center(conv4)
        dec5 = self.dec5(torch.cat([center, conv4], 1))
        dec4 = self.dec4(torch.cat([dec5, conv3], 1))
        dec3 = self.dec3(torch.cat([dec4, conv2], 1))
        dec2 = self.dec2(torch.cat([dec3, conv1], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        #print('input_adjust ', input_adjust.shape, '\ncenter ' , center.shape, '\ndec1: ', dec1.shape, self.final(F.dropout2d(dec0, p=self.dropout_2d).shape))
        return self.final(F.dropout2d(dec0, p=self.dropout_2d))
'''
        center = self.conv4(conv3)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = F.dropout2d(self.dec1(torch.cat([dec2, conv1], 1)), p=self.dropout_2d)
        return self.final(dec1)
        
'''
        

class UNetResNext(nn.Module):
    def __init__(self, encoder_depth, num_classes, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = resnext34()
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = resnext101()
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = resnext152()
            bottom_channel_nr = 2048
         
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnext are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool) ## this pool to delete

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 *2, num_filters * 8, False)
        
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8,   is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        #pool = self.pool(conv5) # deleted pooling
        #center = self.center(pool)
        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


class UNetPNASNet(nn.Module):
    def __init__(self, encoder_depth,  num_classes, num_filters=32, dropout_2d=0.2,
                     pretrained=False, is_deconv=False):
            super().__init__()
            self.num_classes = num_classes
            self.dropout_2d = dropout_2d
            self.encoder = PNASNet5Large()
            bottom_channel_nr = 4320
            self.center = DecoderCenter(bottom_channel_nr, num_filters * 8 *2, num_filters * 8, False)

            self.dec5  =  DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8,   is_deconv)
            self.dec4  = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
            self.dec3  = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
            self.dec2  = DecoderBlockV2(num_filters * 4 * 4, num_filters * 4 * 4, num_filters, is_deconv)
            self.dec1  = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
            self.dec0  = ConvRelu(num_filters, num_filters)
            self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
            features = self.encoder.features(x)
            relued_features = self.encoder.relu(features)
            avg_pooled_features = self.encoder.avg_pool(relued_features)
            center = self.center(avg_pooled_features)
            dec5 = self.dec5(torch.cat([center, avg_pooled_features], 1))
            dec4 = self.dec4(torch.cat([dec5, relued_features], 1))
            dec3 = self.dec3(torch.cat([dec4, features], 1))
            dec2 = self.dec2(dec3)
            dec1 = self.dec1(dec2)
            dec0 = self.dec0(dec1)
            return self.final(F.dropout2d(dec0, p=self.dropout_2d))




class TernausNetV2(nn.Module):
    """Variation of the UNet architecture with InplaceABN encoder."""
    "https://github.com/ternaus/TernausNetV2 by Ternaus 2018"
    def __init__(self, num_classes=1, num_filters=32, is_deconv=False, num_input_channels=3):
        """
        Args:
            num_classes: Number of output classes.
            num_filters:
            is_deconv:
                True: Deconvolution layer is used in the Decoder block.
                False: Upsampling layer is used in the Decoder block.
            num_input_channels: Number of channels in the input images.
        """
        super(TernausNetV2, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        encoder = WiderResNet(structure=[3, 3, 6, 3, 1, 1], classes=0)
        state_dict = torch.load('./modules/wide_resnet38_ipabn_lr_256.pth.tar')['state_dict']
        state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict, strict=False)


        self.conv1 = Sequential(
            OrderedDict([('conv1', nn.Conv2d(num_input_channels, 64, 3, padding=1, bias=False))]))
        self.conv2 = encoder.mod2
        self.conv3 = encoder.mod3
        self.conv4 = encoder.mod4
        self.conv5 = encoder.mod5

        self.center = DecoderBlockTernaus(1024, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec5 = DecoderBlockTernaus(1024 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec4 = DecoderBlockTernaus(512 + num_filters * 8, num_filters * 8, num_filters * 8, is_deconv=is_deconv)
        self.dec3 = DecoderBlockTernaus(256 + num_filters * 8, num_filters * 2, num_filters * 2, is_deconv=is_deconv)
        self.dec2 = DecoderBlockTernaus(128 + num_filters * 2, num_filters * 2, num_filters, is_deconv=is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)



class DecoderBlockTernaus(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=False):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels)
            )

    def forward(self, x):
        return self.block(x)
'''
def AttentionBlock(x,shortcut,i_filters):
    g1 = Conv2D(i_filters,kernel_size = 1)(shortcut) 
    g1 = BatchNormalization()(g1)
    x1 = Conv2D(i_filters,kernel_size = 1)(x) 
    x1 = BatchNormalization()(x1)

    g1_x1 = Add()([g1,x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1,kernel_size = 1)(psi) 
    psi = BatchNormalization()(psi)
    psi = Activation('sigmoid'))(psi)
    x = Multiply()([x,psi])
    return x
'''