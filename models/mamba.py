import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import math
from mamba_ssm.modules.mamba_simple import Mamba
from timm.models.layers import DropPath
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from models.embedding import DataEmbedding


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (1.0 / math.sqrt(x.size(-1)))
        x_normed = x / (rms_x + self.eps)
        return self.weight * x_normed


# 自定义Block实现
class Block(nn.Module):
    def __init__(
            self,
            d_model,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            rms_norm=True,
            residual_in_fp32=False,
            fused_add_norm=False,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32

        self.norm = (RMSNorm if rms_norm else nn.LayerNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        # Mamba层
        self.mixer = Mamba(
            d_model=d_model,
            **(ssm_cfg if ssm_cfg is not None else {}),
            **factory_kwargs,
        )

    def forward(self, hidden_states, residual=None, inference_params=None):
        if residual is None:
            residual = hidden_states

        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        # 应用mixer
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
            hidden_states = hidden_states.to(torch.float32)

        return hidden_states, residual


class TimeSeriesModel(nn.Module):
    def __init__(self,
                 configs,
                 d_model,
                 n_layer,
                 ssm_cfg=None,
                 norm_epsilon=1e-5,
                 rms_norm=True,
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ):
        super(TimeSeriesModel, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = False

        # embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, d_model, configs.embed, configs.freq, configs.dropout)

        # mamba blocks，使用自定义Block
        self.layers = nn.ModuleList([
            Block(
                d_model,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=False,
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(n_layer)
        ])

        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, RMSNorm):
                module.weight.data.fill_(1.0)

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, x_enc, inference_params=None):
        hidden_states = self.enc_embedding(x_enc, None)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if residual is not None:
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
                hidden_states = hidden_states.to(torch.float32)

            hidden_states = hidden_states + residual

        hidden_states = hidden_states.to(dtype=self.norm_f.weight.dtype)
        hidden_states = self.norm_f(hidden_states)

        return hidden_states
