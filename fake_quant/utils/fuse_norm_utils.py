# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import typing
import torch


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    Fuse the RMSNorm/LayerNorm scaling operation into adjacent linear layers.
    
    Mathematical Background:
    ------------------------
    RMSNorm formula: output = weight * x / sqrt(mean(x^2) + eps)
    
    When followed by a linear layer: y = W @ RMSNorm(x)
                                    y = W @ (weight * x / sqrt(mean(x^2) + eps))
    
    We can fuse the weight scaling into W: y = (W * weight) @ (x / sqrt(mean(x^2) + eps))
    
    This function performs: W_new = W_old * layernorm.weight (broadcast multiply)
    
    Benefits:
    ---------
    1. Reduces computation by eliminating the separate weight multiplication
    2. Maintains mathematical equivalence (the RMS normalization still happens at runtime)
    3. Simplifies the graph for quantization
    
    Note on Bias:
    -------------
    If the layernorm has a bias (rare for RMSNorm, common for LayerNorm):
    - We absorb it into the linear layer's bias: b_new = b_old + W_old @ layernorm.bias
    - This is mathematically correct for affine transformations
    
    Args:
        layernorm: The normalization layer (RMSNorm or LayerNorm) to fuse
        linear_layers: The linear layers to fuse the norm weights into
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):
    """
    Fuse layer normalization operations into adjacent linear layers for quantization.
    
    IMPORTANT NOTE ON EMBEDDING MEAN SUBTRACTION:
    ===============================================
    The embedding mean subtraction (W_ - W_.mean) is NOT part of standard RMSNorm fusion.
    
    Background:
    -----------
    - Llama2 and Qwen2 use RMSNorm, which computes: output = weight * x / sqrt(mean(x^2) + eps)
    - RMSNorm does NOT subtract the mean (unlike LayerNorm which does: (x - mean) / sqrt(var))
    - Therefore, mathematically, mean subtraction is not part of RMSNorm fusion
    
    Why the mean subtraction exists:
    ---------------------------------
    This preprocessing step serves a DIFFERENT purpose - it's for QUANTIZATION, not norm fusion:
    
    1. **Outlier Suppression**: Centering embedding vectors around zero reduces extreme values
       that can hurt quantization quality
    
    2. **Improved Quantization Range**: Zero-centered distributions utilize quantization bins
       more uniformly, reducing quantization error
    
    3. **Rotation Stability**: The subsequent basis_change (rotation) operations work better
       on centered data, preserving geometric properties
    
    4. **Activation Distribution**: Helps ensure the first layer receives inputs with more
       stable statistics, reducing activation outliers throughout the network
    
    How the mean subtraction is compensated (canceling mechanism):
    -------------------------------------------------------------
    The model DOES compensate for this distribution shift through the rotation pipeline:
    
    1. **Embeddings are mean-centered**: W_embed = W_embed - mean(W_embed)
    
    2. **Embeddings are rotated**: W_embed_rotated = W_embed @ R1
    
    3. **basis_change_1 is set to learned transformation**: In fuse_basis_to_model(), the first
       layer's basis_change_1 is initialized with torch.eye() but then replaced with
       U_mlp.T @ U_attn, which is a learned transformation (product of orthogonal matrices)
    
    4. **The rotation matrices (R1, U_mlp, U_attn) are optimized** to minimize reconstruction
       error, which implicitly learns to compensate for the mean-centered embeddings
    
    **Summary**: ResQ compensates for the mean subtraction through optimized rotation matrices
    (R1, basis_change_1, etc.) that are learned during the rotation optimization phase.
    The system learns transformations that work well with mean-centered inputs.
    
    Important Considerations:
    -------------------------
    - This changes the embedding distribution from what the model was trained on
    - The rotation optimization (via fuse_basis_to_model) adapts to this changed distribution
    - The learned basis_change transformations provide the compensating mechanism
    - Since we're doing post-training quantization (PTQ), this is an approximation that
      trades off perfect fidelity for better quantization robustness
    - Empirically, this appears to help more than it hurts for the quantization task
    
    Alternative Approaches:
    -----------------------
    - Could skip mean subtraction and rely only on rotation + quantization
    - Could absorb mean subtraction into the first layer's basis_change_1 linear layer
    - Could use learned affine parameters to compensate for the distribution shift
    
    Key Insight:
    -----------
    The mean subtraction CAN be used because:
    1. It's applied BEFORE rotation (rotate_embeddings applies R1 AFTER mean centering)
    2. The rotation matrices (R1, U_mlp, U_attn) are optimized on calibration data
    3. The optimization implicitly learns rotations that work with mean-centered inputs
    4. The basis_change_1 transformation provides additional adaptation capacity
    5. The entire pipeline (mean-center → rotate → basis_change → norm → attention) is
       optimized jointly to minimize reconstruction error on calibration data
    """
    kwargs = {"model": model}

    # Embedding preprocessing for quantization robustness
    # Note: This is NOT standard RMSNorm fusion - see docstring above for explanation
    # if hasattr(model.language_model.model, "embed_tokens"): # for llama-vision
    #     for W in [model.language_model.model.embed_tokens]:
    #         W_ = W.weight.data.double()
    #         W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    #     layers = [layer for layer in model.language_model.model.layers]

    # else:
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.double()
        # Center each embedding vector to improve quantization robustness
        # This is a quantization-motivated preprocessing, not RMSNorm fusion
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = [layer for layer in model.model.layers]

    # Fuse RMSNorm into the adjacent linear blocks
    # This is the ACTUAL norm fusion: multiply layernorm.weight into the linear layer weights
    # After fusion, the norm layers become identity operations (weight = 1)
    for layer in layers:
        # Fuse input_layernorm into attention projections (q, k, v)
        # Original: hidden_states -> input_layernorm -> [q_proj, k_proj, v_proj]
        # After fusion: hidden_states -> [fused_q_proj, fused_k_proj, fused_v_proj]
        # Where fused_W = original_W * layernorm.weight (element-wise along input dim)
        if hasattr(layer, "self_attn"):
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
        elif hasattr(layer, "cross_attn"):
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.cross_attn.q_proj,
                ],
            )
        
        # Fuse post_attention_layernorm into MLP projections (up, gate)
        # Original: hidden_states -> post_attention_layernorm -> [up_proj, gate_proj]
        # After fusion: hidden_states -> [fused_up_proj, fused_gate_proj]
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )

        # Set norm weights to 1.0 (making them identity operations)
        # The normalization effect is now baked into the fused linear layer weights
        # Note: The RMS calculation (division by sqrt(mean(x^2))) is still applied
        # at runtime by the forward pass, but without the learnable weight scaling
        W_norm = layer.post_attention_layernorm.weight.data
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)

    # Fuse the final norm layer into the language model head
    # Original: hidden_states -> model.norm -> lm_head
    # After fusion: hidden_states -> fused_lm_head
    fuse_ln_linear(
        model.model.norm,
        [model.lm_head],
    )
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)
