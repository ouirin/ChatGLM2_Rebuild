import torch
from torch import nn
from quant_tool.triton_ops import triton_kernel

use_triton = True

@torch.no_grad()
def unpack_int4(x, x_scale):

    column_num = x.shape[0] * 2
    group_num, row_num = x_scale.shape
    group_size = column_num // group_num

    shifts = torch.tensor([0, 4]).reshape((1, 2, 1)).type_as(x)

    x = x.unsqueeze(1).repeat((1, 2, 1))
    x = ((x >> shifts) & 0xF).to(torch.int8) - 8

    x = x.reshape((group_num, group_size, row_num)) * x_scale[:, None, :]

    return x.reshape((column_num, row_num))


class QuantLinear(nn.Module):

    def __init__(self, in_features, out_features, group_size=32, bias=True, dtype=None):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer("qweight", torch.empty((in_features//2, out_features), dtype=torch.uint8))
        self.register_buffer("scale", torch.empty((in_features // group_size, out_features), dtype=dtype))
        self.register_buffer('bias', torch.zeros(out_features, dtype=dtype))

    def forward(self, input):

        if use_triton:
            out = triton_kernel(input.flatten(0, -2), self.qweight, self.scale)
            out_shape = (*input.shape[:-1], self.qweight.shape[1])
            out = out.reshape(out_shape)
        else:
            out = input.matmul(unpack_int4(self.qweight, self.scale))

        if self.bias is not None:

            out += self.bias

        return out

    @torch.no_grad()
    def apply_weights_(self, q_weight, scale, bias):

        self.qweight.copy_(q_weight.t())
        self.scale.copy_(scale.t())
        if bias is not None:
            self.bias.copy_(bias)

    def extra_repr(self) -> str:

        return 'in_features={}, out_features={}, group_size={}, weight={}'.format(self.in_features, self.out_features, self.group_size, self.qweight.shape)

    def reset_parameters(self):
        pass
