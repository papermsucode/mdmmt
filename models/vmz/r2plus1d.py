import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import R2Plus1dStem, BasicBlock, Bottleneck


from .utils import _generic_resnet, R2Plus1dStem_Pool, Conv2Plus1D


__all__ = ["r2plus1d_34", "r2plus1d_152"]


def r2plus1d_34(use_pool1=False, **kwargs):
    model = _generic_resnet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model


def r2plus1d_152(use_pool1=True, **kwargs):
    model = _generic_resnet(
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model
