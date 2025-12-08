# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MXFP4 Integration module for RESQ.

This module provides integration utilities to use MXFP4 quantization
within the RESQ fake quantization framework.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from utils.mxfp4_utils import (
    MXFP4Quantizer,
    MXFP4ActQuantizer,
    MXFP4WeightQuantizer,
    MXFP4STE,
    mxfp4_quantize,
)
from utils.quant_utils import ActQuantWrapper, ActQuantizer, WeightQuantizer


class MXFP4ActQuantWrapper(nn.Module):
    """
    Activation Quantization Wrapper using MXFP4 format.
    
    This is a drop-in replacement for ActQuantWrapper that uses MXFP4
    quantization instead of the standard integer quantization.
    """
    
    def __init__(self, module: nn.Linear) -> None:
        super().__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        
        # MXFP4 quantizers
        self.quantizer = MXFP4ActQuantizer()
        self.out_quantizer = MXFP4ActQuantizer()
        
        # Hadamard transform settings (inherited from original)
        self.register_buffer("had_K", None)
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        
        # MXFP4 specific settings
        self.mxfp4_enabled = False
        self.mxfp4_group_size = 32
        self.mxfp4_stochastic = False
    
    def configure_mxfp4(
        self,
        enabled: bool = True,
        group_size: int = 32,
        stochastic: bool = False,
        quantize_output: bool = False,
    ) -> None:
        """Configure MXFP4 quantization settings."""
        self.mxfp4_enabled = enabled
        self.mxfp4_group_size = group_size
        self.mxfp4_stochastic = stochastic
        
        self.quantizer.configure(
            group_size=group_size,
            stochastic=stochastic,
            enabled=enabled,
        )
        
        self.out_quantizer.configure(
            group_size=group_size,
            stochastic=stochastic,
            enabled=quantize_output,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        R1: Optional[torch.Tensor] = None,
        R2: Optional[torch.Tensor] = None,
        transpose: bool = False,
        column_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_dtype = x.dtype
        
        # Apply online Hadamard if configured
        if self.online_full_had and self.had_K is not None:
            from utils import hadamard_utils
            if self.fp32_had:
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        
        # MXFP4 input quantization
        if self.mxfp4_enabled:
            x = self.quantizer(x)
        
        # Apply column reordering if specified
        if column_order is not None:
            x = x[..., column_order]
        
        # Forward through the linear layer
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)
        
        # MXFP4 output quantization
        if self.out_quantizer.enabled:
            x = self.out_quantizer(x)
        
        return x


class HybridQuantizer(nn.Module):
    """
    Hybrid quantizer that can switch between standard INT quantization
    and MXFP4 quantization.
    """
    
    def __init__(self):
        super().__init__()
        self.int_quantizer = ActQuantizer()
        self.mxfp4_quantizer = MXFP4ActQuantizer()
        self.use_mxfp4 = False
    
    def configure(
        self,
        use_mxfp4: bool = False,
        # INT quantizer params
        bits: int = 4,
        groupsize: int = -1,
        sym: bool = False,
        clip_ratio: float = 1.0,
        # MXFP4 params
        mxfp4_group_size: int = 32,
        mxfp4_stochastic: bool = False,
        **kwargs,
    ) -> None:
        """Configure the hybrid quantizer."""
        self.use_mxfp4 = use_mxfp4
        
        if use_mxfp4:
            self.mxfp4_quantizer.configure(
                group_size=mxfp4_group_size,
                stochastic=mxfp4_stochastic,
                enabled=True,
            )
        else:
            self.int_quantizer.configure(
                bits=bits,
                groupsize=groupsize,
                sym=sym,
                clip_ratio=clip_ratio,
                **kwargs,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_mxfp4:
            return self.mxfp4_quantizer(x)
        else:
            return self.int_quantizer(x)
    
    def find_params(self, x: torch.Tensor) -> None:
        if self.use_mxfp4:
            self.mxfp4_quantizer.find_params(x)
        else:
            self.int_quantizer.find_params(x)
    
    def free(self) -> None:
        if self.use_mxfp4:
            self.mxfp4_quantizer.free()
        else:
            self.int_quantizer.free()
    
    @property
    def bits(self) -> int:
        if self.use_mxfp4:
            return 4
        return self.int_quantizer.bits


def replace_quantizers_with_mxfp4(
    model: nn.Module,
    target_modules: Optional[list] = None,
    group_size: int = 32,
    stochastic: bool = False,
) -> None:
    """
    Replace standard quantizers with MXFP4 quantizers in specified modules.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns to target (e.g., ['q_proj', 'k_proj'])
                       If None, replaces all quantizers
        group_size: MXFP4 group size
        stochastic: Use stochastic rounding
    """
    from utils.quant_utils import find_qlayers
    
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    
    for name, layer in qlayers.items():
        # Check if this module should be targeted
        if target_modules is not None:
            should_replace = any(t in name for t in target_modules)
            if not should_replace:
                continue
        
        # Replace input quantizer
        if hasattr(layer, 'quantizer'):
            new_quantizer = MXFP4ActQuantizer()
            new_quantizer.configure(
                group_size=group_size,
                stochastic=stochastic,
                enabled=layer.quantizer.bits < 16,
            )
            layer.quantizer = new_quantizer
        
        # Replace output quantizer
        if hasattr(layer, 'out_quantizer'):
            new_out_quantizer = MXFP4ActQuantizer()
            new_out_quantizer.configure(
                group_size=group_size,
                stochastic=stochastic,
                enabled=layer.out_quantizer.bits < 16,
            )
            layer.out_quantizer = new_out_quantizer


def configure_mxfp4_for_resq(
    model: nn.Module,
    config: Dict[str, Any],
) -> None:
    """
    Configure MXFP4 quantization for a RESQ model based on config.
    
    Config structure:
    {
        'enabled': True,
        'group_size': 32,
        'stochastic': False,
        'targets': {
            'q_proj': {'input': True, 'output': False},
            'k_proj': {'input': True, 'output': False},
            'v_proj': {'input': True, 'output': True},
            'o_proj': {'input': True, 'output': False},
            'up_proj': {'input': True, 'output': False},
            'gate_proj': {'input': True, 'output': False},
            'down_proj': {'input': True, 'output': False},
        }
    }
    """
    if not config.get('enabled', False):
        return
    
    from utils.quant_utils import find_qlayers
    
    group_size = config.get('group_size', 32)
    stochastic = config.get('stochastic', False)
    targets = config.get('targets', {})
    
    qlayers = find_qlayers(model, layers=[ActQuantWrapper])
    
    for name, layer in qlayers.items():
        # Find matching target config
        target_config = None
        for target_name, cfg in targets.items():
            if target_name in name:
                target_config = cfg
                break
        
        if target_config is None:
            continue
        
        # Configure input quantizer
        if target_config.get('input', False):
            new_quantizer = MXFP4ActQuantizer()
            new_quantizer.configure(
                group_size=group_size,
                stochastic=stochastic,
                enabled=True,
            )
            layer.quantizer = new_quantizer
        
        # Configure output quantizer
        if target_config.get('output', False):
            new_out_quantizer = MXFP4ActQuantizer()
            new_out_quantizer.configure(
                group_size=group_size,
                stochastic=stochastic,
                enabled=True,
            )
            layer.out_quantizer = new_out_quantizer