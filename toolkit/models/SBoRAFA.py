#based off https://github.com/catid/dora/blob/main/dora.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Union, List

from optimum.quanto import QBytesTensor, QTensor

from toolkit.network_mixins import ToolkitModuleMixin, ExtractableModuleMixin

if TYPE_CHECKING:
    from toolkit.lora_special import LoRASpecialNetwork

# diffusers specific stuff
LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear'
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T

class SBoRAFAModule(ToolkitModuleMixin, ExtractableModuleMixin, torch.nn.Module):
    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
            network: 'LoRASpecialNetwork' = None,
            use_bias: bool = False,
            **kwargs
    ):
        self.can_merge_in = False
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        ToolkitModuleMixin.__init__(self, network=network)
        torch.nn.Module.__init__(self)
        self.lora_name = lora_name
        self.scalar = torch.tensor(1.0)

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ in CONV_MODULES:
            raise NotImplementedError("Convolutional layers are not supported yet")

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        # self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える eng: treat as constant

        self.multiplier: Union[float, List[float]] = multiplier
        # wrap the original module so it doesn't get weights updated
        self.org_module = [org_module]
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.is_checkpointing = False

        d_out = org_module.out_features
        d_in = org_module.in_features

        std_dev = 1 / torch.sqrt(torch.tensor(self.lora_dim).float())

        self.lora_up = nn.Linear(self.lora_dim, d_out, bias=False)
        self.lora_up.weight.data = torch.zeros_like(self.lora_up.weight.data)
        self.lora_down = nn.Parameter(torch.randint(d_in, (self.lora_dim,), device=self.lora_up.weight.device), requires_grad=False)

        # m = Magnitude column-wise across output dimension
        weight = self.get_orig_weight()
        weight = weight.to(self.lora_up.weight.device, dtype=self.lora_up.weight.dtype)

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward
        # del self.org_module

    def get_orig_weight(self):
        weight = self.org_module[0].weight
        if isinstance(weight, QTensor) or isinstance(weight, QBytesTensor):
            return weight.dequantize().data.detach()
        else:
            return weight.data.detach()

    def get_orig_bias(self):
        if hasattr(self.org_module[0], 'bias') and self.org_module[0].bias is not None:
            return self.org_module[0].bias.data.detach()
        return None
    
    def _call_forward(self, x):
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0  # added to original forward

        if hasattr(self, 'lora_mid') and self.lora_mid is not None:
            lx = self.lora_mid(x[..., self.lora_down])
        else:
            try:
                lx = x[..., self.lora_down]
            except RuntimeError as e:
                print(f"Error in {self.__class__.__name__} lora_down")
                print(e)

        if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
            lx = self.dropout(lx)

        # normal dropout
        elif self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.rank_dropout > 0 and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale
        
        lx = self.lora_up(lx)

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        return lx * scale