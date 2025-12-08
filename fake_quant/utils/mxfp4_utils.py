# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
MXFP4 (Microscaling FP4) Quantization Utilities.

MXFP4 Format:
- Group size: 32 elements share a common scale
- Shared scale: E8M0 format (8-bit exponent, no mantissa, bias=127)
- Elements: E2M1 format (1 sign bit, 2-bit exponent, 1-bit mantissa)

E2M1 representable values (with sign):
±{0, 0.5, 1, 1.5, 2, 3, 4, 6}
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


# E2M1 format constants
# E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
# Exponent bias = 1
# Representable positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MAX = 6.0
E2M1_MIN_POSITIVE = 0.5

# E8M0 format constants (for shared scale)
# E8M0: 8 exponent bits, no mantissa, bias = 127
E8M0_BIAS = 127
E8M0_MAX_EXP = 127  # Max exponent value (2^127)
E8M0_MIN_EXP = -126  # Min exponent value (2^-126)


def get_e2m1_lookup_table(device: torch.device = None) -> torch.Tensor:
    """
    Get the E2M1 lookup table for quantization.
    Returns tensor of shape (16,) with all E2M1 representable values (including negative).
    """
    # Positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # Negative values: -0, -0.5, -1, -1.5, -2, -3, -4, -6
    positive = E2M1_VALUES.clone()
    negative = -E2M1_VALUES.clone()
    # Combine: [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]
    # But -0 = 0, so we have 15 unique values
    lookup = torch.cat([positive, negative[1:]])  # Skip -0
    if device is not None:
        lookup = lookup.to(device)
    return lookup


def compute_e8m0_scale(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Compute E8M0 shared scale for each group.
    
    E8M0 format: 2^exponent where exponent is an 8-bit integer with bias 127.
    The scale is chosen so that the maximum absolute value in the group
    maps to the maximum E2M1 value (6.0).
    
    Args:
        x: Input tensor of any shape, last dimension will be grouped
        group_size: Number of elements per group (default: 32)
    
    Returns:
        Scale tensor with shape [..., num_groups]
    """
    # Reshape to group elements
    orig_shape = x.shape
    if orig_shape[-1] % group_size != 0:
        raise ValueError(f"Last dimension {orig_shape[-1]} must be divisible by group_size {group_size}")
    
    num_groups = orig_shape[-1] // group_size
    x_grouped = x.reshape(*orig_shape[:-1], num_groups, group_size)
    
    # Find max absolute value in each group
    amax = x_grouped.abs().amax(dim=-1)  # [..., num_groups]
    
    # Compute the ideal scale: scale = amax / E2M1_MAX
    # Then quantize scale to E8M0 format (power of 2)
    # E8M0: scale = 2^(exp - 127), where exp is 8-bit unsigned int
    
    # Avoid log2(0) by clamping
    amax_clamped = amax.clamp(min=1e-38)
    
    # Compute exponent: we want scale * E2M1_MAX >= amax
    # scale = 2^exp, so exp = ceil(log2(amax / E2M1_MAX))
    ideal_exp = torch.ceil(torch.log2(amax_clamped / E2M1_MAX))
    
    # Clamp exponent to E8M0 range
    exp_clamped = ideal_exp.clamp(min=E8M0_MIN_EXP, max=E8M0_MAX_EXP)
    
    # Compute scale = 2^exp
    scale = torch.pow(2.0, exp_clamped)
    
    # Handle zeros: if amax is 0, set scale to 1 (arbitrary, values are all 0)
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    
    return scale


def quantize_to_e2m1(x_scaled: torch.Tensor) -> torch.Tensor:
    """
    Quantize scaled values to E2M1 format using nearest rounding.
    
    Args:
        x_scaled: Input tensor scaled to E2M1 range [-6, 6]
    
    Returns:
        Quantized tensor with values in E2M1 set
    """
    device = x_scaled.device
    dtype = x_scaled.dtype
    
    # E2M1 positive values
    e2m1_vals = E2M1_VALUES.to(device=device, dtype=dtype)
    
    # Get sign and absolute value
    sign = torch.sign(x_scaled)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # Handle 0 -> +0
    abs_x = x_scaled.abs()
    
    # Clamp to valid range
    abs_x = abs_x.clamp(max=E2M1_MAX)
    
    # Find nearest E2M1 value using broadcasting
    # Shape: [..., 1] vs [8] -> [..., 8]
    abs_x_expanded = abs_x.unsqueeze(-1)
    distances = (abs_x_expanded - e2m1_vals).abs()
    nearest_idx = distances.argmin(dim=-1)
    
    # Get quantized absolute value
    quantized_abs = e2m1_vals[nearest_idx]
    
    # Apply sign
    quantized = sign * quantized_abs
    
    return quantized


def stochastic_round_to_e2m1(x_scaled: torch.Tensor) -> torch.Tensor:
    """
    Quantize scaled values to E2M1 format using stochastic rounding.
    
    Args:
        x_scaled: Input tensor scaled to E2M1 range [-6, 6]
    
    Returns:
        Quantized tensor with values in E2M1 set
    """
    device = x_scaled.device
    dtype = x_scaled.dtype
    
    e2m1_vals = E2M1_VALUES.to(device=device, dtype=dtype)
    
    sign = torch.sign(x_scaled)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    abs_x = x_scaled.abs().clamp(max=E2M1_MAX)
    
    # Find the two nearest E2M1 values (floor and ceil in E2M1 space)
    abs_x_expanded = abs_x.unsqueeze(-1)
    
    # Find values <= abs_x (floor)
    mask_floor = e2m1_vals <= abs_x_expanded
    floor_vals = torch.where(mask_floor, e2m1_vals, torch.zeros_like(e2m1_vals))
    floor_val = floor_vals.amax(dim=-1)
    
    # Find values >= abs_x (ceil)
    mask_ceil = e2m1_vals >= abs_x_expanded
    ceil_vals = torch.where(mask_ceil, e2m1_vals, torch.full_like(e2m1_vals, float('inf')))
    ceil_val = ceil_vals.amin(dim=-1)
    ceil_val = torch.where(ceil_val == float('inf'), floor_val, ceil_val)
    
    # Compute probability of rounding up
    range_size = ceil_val - floor_val
    prob_ceil = torch.where(
        range_size > 0,
        (abs_x - floor_val) / range_size,
        torch.zeros_like(abs_x)
    )
    
    # Stochastic rounding
    rand = torch.rand_like(abs_x)
    quantized_abs = torch.where(rand < prob_ceil, ceil_val, floor_val)
    
    return sign * quantized_abs


class MXFP4Quantizer(nn.Module):
    """
    MXFP4 Quantizer for microscaling FP4 quantization.
    
    Format:
    - Group size: 32 elements
    - Shared scale: E8M0 (8-bit exponent, bias=127)
    - Elements: E2M1 (1 sign, 2 exp, 1 mantissa)
    """
    
    def __init__(
        self,
        group_size: int = 32,
        stochastic: bool = False,
    ):
        super().__init__()
        self.group_size = group_size
        self.stochastic = stochastic
        
        # Register E2M1 lookup table as buffer
        self.register_buffer('e2m1_values', E2M1_VALUES.clone())
        
        # Buffers for storing computed scales (for analysis/debugging)
        self.register_buffer('last_scale', torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor to MXFP4 format and dequantize back.
        
        Args:
            x: Input tensor, last dimension must be divisible by group_size
        
        Returns:
            Fake-quantized tensor (same shape as input)
        """
        return self.quantize_dequantize(x)
    
    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform fake quantization: quantize to MXFP4 then dequantize.
        """
        orig_shape = x.shape
        orig_dtype = x.dtype
        
        # Handle case where last dim is not divisible by group_size
        last_dim = orig_shape[-1]
        if last_dim % self.group_size != 0:
            # Pad to make divisible
            pad_size = self.group_size - (last_dim % self.group_size)
            x = torch.nn.functional.pad(x, (0, pad_size), value=0)
            padded = True
        else:
            pad_size = 0
            padded = False
        
        # Work in float32 for precision
        x = x.float()
        
        # Compute E8M0 scale for each group
        scale = compute_e8m0_scale(x, self.group_size)  # [..., num_groups]
        self.last_scale = scale.detach()
        
        # Reshape for group-wise operations
        num_groups = x.shape[-1] // self.group_size
        x_grouped = x.reshape(*x.shape[:-1], num_groups, self.group_size)
        
        # Scale values to E2M1 range
        scale_expanded = scale.unsqueeze(-1)  # [..., num_groups, 1]
        x_scaled = x_grouped / scale_expanded
        
        # Quantize to E2M1
        if self.stochastic:
            x_quantized = stochastic_round_to_e2m1(x_scaled)
        else:
            x_quantized = quantize_to_e2m1(x_scaled)
        
        # Dequantize: multiply by scale
        x_dequantized = x_quantized * scale_expanded
        
        # Reshape back
        x_dequantized = x_dequantized.reshape(*x.shape[:-1], -1)
        
        # Remove padding if added
        if padded:
            x_dequantized = x_dequantized[..., :last_dim]
        
        return x_dequantized.to(orig_dtype)
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize input tensor to MXFP4 format.
        
        Returns:
            Tuple of (quantized_values, scales)
            - quantized_values: E2M1 quantized values (not packed)
            - scales: E8M0 scales for each group
        """
        orig_shape = x.shape
        x = x.float()
        
        # Compute E8M0 scale
        scale = compute_e8m0_scale(x, self.group_size)
        
        # Reshape and scale
        num_groups = x.shape[-1] // self.group_size
        x_grouped = x.reshape(*x.shape[:-1], num_groups, self.group_size)
        scale_expanded = scale.unsqueeze(-1)
        x_scaled = x_grouped / scale_expanded
        
        # Quantize
        if self.stochastic:
            x_quantized = stochastic_round_to_e2m1(x_scaled)
        else:
            x_quantized = quantize_to_e2m1(x_scaled)
        
        return x_quantized, scale


class MXFP4STE(torch.autograd.Function):
    """
    Straight-Through Estimator for MXFP4 quantization.
    Forward: quantize to MXFP4
    Backward: pass gradients through unchanged
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int = 32, stochastic: bool = False) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        
        # Handle padding
        last_dim = orig_shape[-1]
        if last_dim % group_size != 0:
            pad_size = group_size - (last_dim % group_size)
            x = torch.nn.functional.pad(x, (0, pad_size), value=0)
        else:
            pad_size = 0
        
        x = x.float()
        
        # Compute scale
        scale = compute_e8m0_scale(x, group_size)
        
        # Reshape and scale
        num_groups = x.shape[-1] // group_size
        x_grouped = x.reshape(*x.shape[:-1], num_groups, group_size)
        scale_expanded = scale.unsqueeze(-1)
        x_scaled = x_grouped / scale_expanded
        
        # Quantize
        if stochastic:
            x_quantized = stochastic_round_to_e2m1(x_scaled)
        else:
            x_quantized = quantize_to_e2m1(x_scaled)
        
        # Dequantize
        x_dequantized = x_quantized * scale_expanded
        x_dequantized = x_dequantized.reshape(*x.shape[:-1], -1)
        
        # Remove padding
        if pad_size > 0:
            x_dequantized = x_dequantized[..., :last_dim]
        
        return x_dequantized.to(orig_dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through: pass gradients unchanged
        return grad_output, None, None


class MXFP4ActQuantizer(nn.Module):
    """
    MXFP4 Activation Quantizer compatible with RESQ's ActQuantizer interface.
    
    This quantizer can replace or work alongside the existing ActQuantizer
    for MXFP4 quantization of activations.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.group_size = 32
        self.stochastic = False
        self.enabled = False
        self.bits = 4  # For compatibility
        
        # Store last computed scale for debugging
        self.register_buffer('last_scale', torch.zeros(1))
        self.register_buffer('e2m1_values', E2M1_VALUES.clone())
    
    def configure(
        self,
        group_size: int = 32,
        stochastic: bool = False,
        enabled: bool = True,
    ) -> None:
        """Configure the MXFP4 quantizer."""
        self.group_size = group_size
        self.stochastic = stochastic
        self.enabled = enabled
        self.bits = 4 if enabled else 16
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize activation using MXFP4."""
        if not self.enabled:
            return x
        return MXFP4STE.apply(x, self.group_size, self.stochastic)
    
    def find_params(self, x: torch.Tensor) -> None:
        """Compute and store scale (for compatibility with ActQuantizer interface)."""
        if not self.enabled:
            return
        try:
            self.last_scale = compute_e8m0_scale(x, self.group_size)
        except ValueError:
            # Handle case where dimension is not divisible by group_size
            pass
    
    def free(self) -> None:
        """Free stored parameters."""
        self.last_scale = torch.zeros(1, device=self.last_scale.device)


class MXFP4WeightQuantizer(nn.Module):
    """
    MXFP4 Weight Quantizer compatible with RESQ's WeightQuantizer interface.
    """
    
    def __init__(self, shape: int = 1) -> None:
        super().__init__()
        self.group_size = 32
        self.stochastic = False
        self.bits = 4
        
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('e2m1_values', E2M1_VALUES.clone())
    
    def configure(
        self,
        group_size: int = 32,
        stochastic: bool = False,
        **kwargs  # Accept additional args for compatibility
    ) -> None:
        """Configure the MXFP4 weight quantizer."""
        self.group_size = group_size
        self.stochastic = stochastic
        self.bits = 4
    
    def find_params(self, x: torch.Tensor) -> None:
        """Compute and store scale for weight quantization."""
        if self.bits >= 16:
            return
        self.scale = compute_e8m0_scale(x, self.group_size)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize weights using MXFP4."""
        if self.bits >= 16:
            return x
        return MXFP4STE.apply(x, self.group_size, self.stochastic)
    
    def enabled(self) -> bool:
        return self.bits < 16
    
    def ready(self) -> bool:
        return torch.any(self.scale != 0)


# Convenience functions for direct use
def mxfp4_quantize(x: torch.Tensor, group_size: int = 32, stochastic: bool = False) -> torch.Tensor:
    """
    Quantize tensor to MXFP4 format and dequantize.
    
    Args:
        x: Input tensor
        group_size: Elements per group (default: 32)
        stochastic: Use stochastic rounding (default: False)
    
    Returns:
        Fake-quantized tensor
    """
    return MXFP4STE.apply(x, group_size, stochastic)


def mxfp4_quantize_weight(weight: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Quantize weight tensor to MXFP4 format.
    
    For weights, we typically use deterministic rounding.
    """
    return mxfp4_quantize(weight, group_size, stochastic=False)


def mxfp4_quantize_activation(activation: torch.Tensor, group_size: int = 32, stochastic: bool = False) -> torch.Tensor:
    """
    Quantize activation tensor to MXFP4 format.
    
    For activations, stochastic rounding can help reduce bias.
    """
    return mxfp4_quantize(activation, group_size, stochastic=stochastic)