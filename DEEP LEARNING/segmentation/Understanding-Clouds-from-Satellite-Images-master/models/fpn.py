# https://raw.githubusercontent.com/BloodAxe/Catalyst-CamVid-Segmentation-Example/master/models/fpn.py

from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules import decoders as D
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.fpn import *
from torch import nn
from torch.nn import functional as F


class DoubleConvRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            out_filters, out_filters, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        return x


class DoubleConvBNRelu(nn.Module):
    def __init__(self, in_dec_filters: int, out_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_dec_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(
            out_filters, out_filters, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        return x


class FPNSegmentationModel(nn.Module):
    def __init__(
        self, encoder: E.EncoderModule, num_classes: int, fpn_features=128, dropout=0.25
    ):
        super().__init__()

        decoder = D.FPNDecoder(
            features=encoder.output_filters,
            prediction_block=DoubleConvBNRelu,
            bottleneck=FPNBottleneckBlockBN,
            fpn_features=fpn_features,
            prediction_features=fpn_features,
        )

        self.encoder = encoder
        self.decoder = decoder

        self.fpn_fuse = FPNFuse()
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        output_features = sum(self.decoder.output_filters)

        # Final Classifier
        self.final_decoder = DoubleConvBNRelu(output_features, fpn_features // 2)
        self.logits = nn.Conv2d(fpn_features // 2, num_classes, kernel_size=1)

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)

        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features)

        features = self.fpn_fuse(dec_features)
        features = self.dropout(features)
        features = self.final_decoder(features)

        logits = self.logits(features)
        logits = F.interpolate(
            logits, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True
        )

        logits = unpad_image_tensor(logits, pad)
        return logits

    def set_encoder_training_enabled(self, enabled):
        self.encoder.set_trainable(enabled)


def resnet34_fpn(num_classes=1, fpn_features=128):
    encoder = E.Resnet34Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)


def resnext50_fpn(num_classes=1, fpn_features=128):
    encoder = E.SEResNeXt50Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features, dropout=0.5)


def resnet101_fpn(num_classes=1, fpn_features=256):
    encoder = E.Resnet101Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)


def effnetB4_fpn(num_classes=1, fpn_features=128):
    encoder = E.EfficientNetB4Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)


def effnetB7_fpn(num_classes=1, fpn_features=256):
    encoder = E.EfficientNetB7Encoder()
    return FPNSegmentationModel(encoder, num_classes, fpn_features)
