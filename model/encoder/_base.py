import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict



class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]


    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

