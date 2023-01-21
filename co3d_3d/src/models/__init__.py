import logging

import gin

from .mink.dgcnn import DGCNN_cls
from .mink.fcnn import MinkowskiFCNN, MinkowskiSplatFCNN
from .mink.pointnet import MinkowskiPointNet
from .mink.res16unet import *
from .mink.resnet import *
from .mink.resunet import *
from .paconv.PointNet_PAConv import PAConvPointNet

logger = logging.getLogger(__name__)

MODELS = {var.__name__: var for var in globals() if isinstance(var, nn.Module)}


@gin.configurable
def get_model(name: str, in_channel, out_channel, sparse=None):
    return globals()[name](in_channel=in_channel, out_channel=out_channel)
