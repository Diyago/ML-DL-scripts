import pretrainedmodels
import segmentation_models_pytorch as smp
from .fpn import resnet34_fpn, effnetB4_fpn
import torch.nn as nn
import torchvision
import torch
from typing import Optional, Type


def get_model(
    model_type: str = "Unet",
    encoder: str = "Resnet18",
    encoder_weights: str = "imagenet",
    activation: str = None,
    n_classes: int = 4,
    task: str = "segmentation",
    source: str = "pretrainedmodels",
    head: str = "simple",
):
    """
    Get model for training or inference.

    Returns loaded models, which is ready to be used.

    Args:
        model_type: segmentation model architecture
        encoder: encoder of the model
        encoder_weights: pre-trained weights to use
        activation: activation function for the output layer
        n_classes: number of classes in the output layer
        task: segmentation or classification
        source: source of model for classification
        head: simply change number of outputs or use better output head

    Returns:

    """
    if task == "segmentation":
        if model_type == "Unet":
            model = smp.Unet(
                # attention_type='scse',
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
            )

        elif model_type == "Linknet":
            model = smp.Linknet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
            )

        elif model_type == "FPN":
            model = smp.FPN(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                classes=n_classes,
                activation=activation,
            )

        elif model_type == "resnet34_fpn":
            model = resnet34_fpn(num_classes=n_classes, fpn_features=128)

        elif model_type == "effnetB4_fpn":
            model = effnetB4_fpn(num_classes=n_classes, fpn_features=128)

        else:
            model = None

    elif task == "classification":
        if source == "pretrainedmodels":
            model_fn = pretrainedmodels.__dict__[encoder]
            model = model_fn(num_classes=1000, pretrained=encoder_weights)
        elif source == "torchvision":
            model = torchvision.models.__dict__[encoder](pretrained=encoder_weights)

        if head == "simple":
            model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
        else:
            model = Net(net=model)

    return model


class Flatten(nn.Module):
    """
    Simple class for flattening layer.

    """

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Net(nn.Module):
    def __init__(self, num_classes: int = 4, p: float = 0.2, net=None) -> None:
        """
        Custom head architecture

        Args:
            num_classes: number of classes
            p: dropout probability
            net: original model
        """
        super().__init__()
        modules = list(net.children())[:-1]
        n_feats = list(net.children())[-1].in_features
        # add custom head
        modules += [
            nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(81536),
                nn.Dropout(p),
                nn.Linear(81536, n_feats),
                nn.Linear(n_feats, num_classes),
                nn.Sigmoid(),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, size: Optional[int] = None):
        "Output will be 2*size or 2 if size is None"
        super().__init__()
        size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x: Type[torch.Tensor]) -> Type[torch.Tensor]:
        return torch.cat([self.mp(x), self.ap(x)], 1)
