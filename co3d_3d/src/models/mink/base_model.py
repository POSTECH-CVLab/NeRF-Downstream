import MinkowskiEngine as ME
import torch
from co3d_3d.src.models.interface import InputInterface


class MinkowskiBaseModel(ME.MinkowskiNetwork, InputInterface):
    def __init__(self, dimension=3):
        ME.MinkowskiNetwork.__init__(self, dimension)

    def process_input(self, batch):
        return ME.TensorField(
            coordinates=batch["coordinates"], features=batch["features"]
        )
