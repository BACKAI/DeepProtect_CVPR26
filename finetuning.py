import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from networks import FullyConnectedLayer

class LoraInjectLinear(nn.Module):
    def __init__(
            self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        self.r = r
        self.linear = FullyConnectedLayer(in_features, out_features, bias, bias_init=1)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1/r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

def find_module(
        model,
        search_class
):
    middle = [16, 32]
    for fullname, module in model.named_modules():
        for i in middle:
            for j in ['conv0', 'conv1', 'torgb']:
                if fullname == f'synthesis.b{i}.{j}.affine':
                    *path, name = fullname.split('.')
                    parent = model._modules[path[0]]

                    for i in path[1:]:
                        parent = parent._modules[i]
                
                    yield parent, name, module


def inject_lora(
        model: nn.Module,
        target_replace_module = FullyConnectedLayer,
        r: int = 8,
        dropout_p = 0.1,
        scale = 1.0
):
    
    for _module, name, _child_module in find_module(model, target_replace_module):
        weight = _child_module.weight
        bias = _child_module.bias
        tmp = LoraInjectLinear(
            _child_module.weight.shape[1],
            _child_module.weight.shape[0],
            _child_module.bias is not None,
            r = r
        )
        tmp.linear.weight = weight
        if bias is not None:
            tmp.linear.bias = bias

        tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        if bias is not None:
            tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

        _module._modules[name] = tmp

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True


