import gc
import torch
from torch import nn
from quant_tool.quant_linear import QuantLinear


def bind_quantizer(model, linear_bit=None, linear_group=None):

    layers_with_path = find_layers(module=model, target=nn.Linear)

    qlayers = {}

    for path, layer in layers_with_path.items():

        qlayers[path] = SimpleQuantizer(layer=layer, path=path, bit=linear_bit, group_size=linear_group)

    return qlayers


def replace_linear(model, qlayers):

    for path, layer in qlayers.items():

        cover_layer(model, path, layer.get_quantized_linear())

        print(f"{path} - finished")


class SimpleQuantizer():

    def __init__(self, layer, path=None, bit=None, group_size=None):

        self.layer = layer
        self.path = path
        self.bit = bit
        self.group_size = group_size
        self.maxq = torch.tensor(2 ** (bit-1) - 1)
        self.n_rows, self.n_columns = layer.weight.shape

    @torch.no_grad()
    def quantize(self):

        scale_list = []

        for i in range(0, self.n_columns, self.group_size):

            w = self.layer.weight[:, i:i + self.group_size]

            w_max, _ = torch.max(w.abs(), dim=1, keepdim=True)
            scale = torch.clamp(w_max / self.maxq, min=1e-10)
            q_weight = torch.clamp(torch.round(w / scale), -self.maxq, self.maxq)

            self.layer.weight[:, i:i + self.group_size] = q_weight
            scale_list.append(scale)

        scale = torch.cat(scale_list, dim=1)

        q_weight = (self.layer.weight + 8).to(torch.uint8)
        q_weight = (q_weight[:, ::2]) | ((q_weight[:, 1::2]) << 4)

        return q_weight, scale

    @torch.no_grad()
    def get_quantized_linear(self):

        q_weight, scale = self.quantize()

        qlinear = QuantLinear(in_features=self.layer.in_features, out_features=self.layer.out_features, group_size=self.group_size)

        qlinear.apply_weights_(q_weight, scale, self.layer.bias)

        del self.layer
        gc.collect()

        return qlinear


def find_layers(module=None, target=None, name=''):

    if isinstance(module, target):

        return {name: module}

    result = {}

    for child_name, child in module.named_children():

        result.update(find_layers(module=child, target=target, name = name + '.' + child_name if name != '' else child_name))

    return result


def cover_layer(module=None, path=None, new_layer=None):

    parent = module
    path_list = path.split('.')
    path_list_len = len(path_list)

    counter = 1

    for child_name in path_list:

        if counter == path_list_len:

            setattr(parent, child_name, new_layer)

        else:

            parent = getattr(parent, child_name)
            counter += 1