import torch
import torch.nn as nn
from MinkowskiEngine import SparseTensor, TensorField


class MinkowskiLayerNorm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        affine=True,
    ):
        super(MinkowskiLayerNorm, self).__init__()
        self.ln = nn.LayerNorm(num_features, eps, elementwise_affine=affine)

    def forward(self, input):
        output = self.ln(input.F)
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    def __repr__(self):
        s = "({}, eps={}, affine={})".format(
            self.ln.normalized_shape,
            self.ln.eps,
            self.ln.elementwise_affine,
        )
        return self.__class__.__name__ + s
