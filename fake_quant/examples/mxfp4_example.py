"""
Example script demonstrating MXFP4 quantization usage.
"""

import torch
from utils.mxfp4_utils import (
    MXFP4Quantizer,
    MXFP4ActQuantizer,
    MXFP4WeightQuantizer,
    mxfp4_quantize,
    compute_e8m0_scale,
    E2M1_VALUES,
)


def demo_basic_mxfp4():
    """Demonstrate basic MXFP4 quantization."""
    print("=" * 60)
    print("Basic MXFP4 Quantization Demo")
    print("=" * 60)
    
    # Create a sample tensor
    x = torch.randn(2, 4, 128)  # [batch, seq, hidden]
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    # Quantize using MXFP4
    x_quant = mxfp4_quantize(x, group_size=32)
    print(f"Quantized shape: {x_quant.shape}")
    print(f"Quantized range: [{x_quant.min():.4f}, {x_quant.max():.4f}]")
    
    # Compute error
    mse = ((x - x_quant) ** 2).mean()
    print(f"MSE: {mse:.6f}")
    print()


def demo_quantizer_module():
    """Demonstrate MXFP4 quantizer module."""
    print("=" * 60)
    print("MXFP4 Quantizer Module Demo")
    print("=" * 60)
    
    # Create quantizer
    quantizer = MXFP4Quantizer(group_size=32, stochastic=False)
    
    # Create sample data
    x = torch.randn(1, 8, 256)
    print(f"Input shape: {x.shape}")
    
    # Quantize
    x_quant = quantizer(x)
    print(f"Output shape: {x_quant.shape}")
    
    # Check scales
    scale = quantizer.last_scale
    print(f"Scale shape: {scale.shape}")
    print(f"Scale range: [{scale.min():.4f}, {scale.max():.4f}]")
    print()


def demo_act_quantizer():
    """Demonstrate MXFP4 activation quantizer (RESQ-compatible)."""
    print("=" * 60)
    print("MXFP4 Act Quantizer (RESQ-compatible) Demo")
    print("=" * 60)
    
    # Create quantizer
    act_quant = MXFP4ActQuantizer()
    act_quant.configure(group_size=32, stochastic=False, enabled=True)
    
    # Simulate activation
    x = torch.randn(2, 16, 512)
    print(f"Input shape: {x.shape}")
    
    # Find params and quantize
    act_quant.find_params(x)
    x_quant = act_quant(x)
    
    print(f"Output shape: {x_quant.shape}")
    print(f"MSE: {((x - x_quant) ** 2).mean():.6f}")
    print()


def demo_weight_quantizer():
    """Demonstrate MXFP4 weight quantizer."""
    print("=" * 60)
    print("MXFP4 Weight Quantizer Demo")
    print("=" * 60)
    
    # Create a weight matrix
    weight = torch.randn(512, 2048)  # [out_features, in_features]
    print(f"Weight shape: {weight.shape}")
    
    # Create quantizer
    w_quant = MXFP4WeightQuantizer()
    w_quant.configure(group_size=32)
    
    # Quantize
    w_quant.find_params(weight)
    weight_quant = w_quant.quantize(weight)
    
    print(f"Quantized weight shape: {weight_quant.shape}")
    print(f"MSE: {((weight - weight_quant) ** 2).mean():.6f}")
    print()


def demo_e2m1_values():
    """Show E2M1 representable values."""
    print("=" * 60)
    print("E2M1 Format Values")
    print("=" * 60)
    print(f"Positive E2M1 values: {E2M1_VALUES.tolist()}")
    print(f"Full range: {(-E2M1_VALUES).flip(0).tolist()[:-1] + E2M1_VALUES.tolist()}")
    print()


def demo_gradient_flow():
    """Demonstrate gradient flow through MXFP4 (STE)."""
    print("=" * 60)
    print("Gradient Flow Demo (STE)")
    print("=" * 60)
    
    # Create input with gradient tracking
    x = torch.randn(2, 4, 64, requires_grad=True)
    
    # Forward through MXFP4
    x_quant = mxfp4_quantize(x, group_size=32)
    
    # Compute loss and backward
    loss = x_quant.sum()
    loss.backward()
    
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Gradient norm: {x.grad.norm():.4f}")
    print("Gradient flows through unchanged (STE)")
    print()


if __name__ == "__main__":
    demo_e2m1_values()
    demo_basic_mxfp4()
    demo_quantizer_module()
    demo_act_quantizer()
    demo_weight_quantizer()
    demo_gradient_flow()