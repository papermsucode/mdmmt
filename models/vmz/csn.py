import warnings

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import BasicStem, BasicBlock, Bottleneck

from .utils import _generic_resnet, Conv3DDepthwise, BasicStem_Pool, IPConv3DDepthwise


__all__ = ["ir_csn_152", "ip_csn_152"]


def ir_csn_152(use_pool1=True, **kwargs):
    model = _generic_resnet(
        block=Bottleneck,
        conv_makers=[Conv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool if use_pool1 else BasicStem,
        **kwargs)
    return model


def ip_csn_152(use_pool1=True, **kwargs):
    model = _generic_resnet(
        block=Bottleneck,
        conv_makers=[IPConv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool if use_pool1 else BasicStem,
        **kwargs)
    return model
