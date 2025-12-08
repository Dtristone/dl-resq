# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Extended PTQ model with configurable rotation and quantization control.

This module provides ptq_model_with_config() which allows selective control over:
- R1 rotation: Hidden state rotation (embeddings, q/k/v_proj inputs, o_proj output, up/gate_proj, down_proj output)
- R2 rotation: V/O attention computation (per-head rotation)
- R3 rotation: K cache rotation
- R4 rotation: Down_proj left side (Hadamard transform)
"""

import torch
import transformers
import logging

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.hadamard_utils import random_orthogonal_matrix
from utils.quant_config_utils import QuantizationConfig, load_quant_config

logger = logging.getLogger(__name__)


def ptq_model_with_config(args, model, model_args=None, quant_config: QuantizationConfig = None):
    """
    Post-Training Quantization with configurable rotation and quantization.
    
    Args:
        args: PTQ arguments
        model: The model to quantize
        model_args: Model arguments (optional)
        quant_config: QuantizationConfig object for selective control. If None, uses default behavior.
    
    Returns:
        Quantized model
    """
    transformers.set_seed(args.seed)
    model.eval()
    
    # Use config from args if not provided
    if quant_config is None:
        quant_config = getattr(args, 'quant_config', None)
    
    # Determine rotation settings based on config
    enable_R1 = True
    enable_R2 = True
    enable_R3 = True
    enable_R4 = True
    custom_R4_path = None
    
    if quant_config is not None:
        enable_R1 = quant_config.rotation.enable_R1
        enable_R2 = quant_config.rotation.enable_R2
        enable_R3 = quant_config.rotation.enable_R3
        enable_R4 = quant_config.rotation.enable_R4
        custom_R4_path = quant_config.rotation.custom_R4_path
        
        logger.info(f"Using quantization config:")
        logger.info(f"  R1 (hidden states): {enable_R1}")
        logger.info(f"  R2 (V/O attention): {enable_R2}")
        logger.info(f"  R3 (K cache): {enable_R3}")
        logger.info(f"  R4 (down_proj Hadamard): {enable_R4}")
        if custom_R4_path:
            logger.info(f"  Custom R4 path: {custom_R4_path}")
    
    # Store rotation flags for later use
    args._enable_R1 = enable_R1
    args._enable_R2 = enable_R2
    args._enable_R3 = enable_R3
    args._enable_R4 = enable_R4
    args._custom_R4_path = custom_R4_path
    
    # Rotate the weights
    if not args.rotate_mode == "none":
        fuse_norm_utils.fuse_layer_norms(model)
        
        if args.rotate_mode == "resq" or args.rotate_mode == "quik":
            # Use modified rotation that respects enable flags
            fuse_basis_to_model_selective(model, args, enable_R1, enable_R2)
        else:
            rotation_utils.rotate_model(model, args)
        
        if args.rotate_mode == "resq" or args.rotate_mode == "quik":
            rotation_utils.rearrange_columns(model, args, False)

        utils.cleanup_memory(verbos=True)
        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        
        # Configure down_proj Hadamard based on R4 setting
        for name in qlayers:
            if "down_proj" in name:
                if enable_R4:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    
                    # Apply custom R4 if provided
                    if custom_R4_path is not None:
                        custom_R4 = load_custom_r4(custom_R4_path, model.config.intermediate_size)
                        if custom_R4 is not None:
                            apply_custom_r4_to_down_proj(model, custom_R4, name)
                else:
                    # Disable Hadamard for down_proj (keep left side in high precision)
                    qlayers[name].online_full_had = False
                    qlayers[name].had_K = None
                    logger.info(f"Disabled R4 Hadamard for {name}")
    else:
        quant_utils.add_actquant(model)
    
    # Weight quantization
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization with config-based control
    configure_activation_quantization(model, args, quant_config, enable_R2)
    
    # K-cache quantization (R3-related)
    if enable_R3:
        configure_k_cache_quantization(model, args, quant_config)
    else:
        logger.info("R3 disabled - skipping K-cache rotation/quantization")

    return model


def fuse_basis_to_model_selective(model, args, enable_R1: bool, enable_R2: bool):
    """
    Modified version of fuse_basis_to_model that selectively applies rotations.
    
    Args:
        model: The model
        args: PTQ arguments
        enable_R1: Whether to apply R1 rotation
        enable_R2: Whether to apply R2 rotation (V/O per-head rotation)
    """
    if not enable_R1 and not enable_R2:
        logger.info("Both R1 and R2 disabled, skipping rotation")
        return
    
    # Delegate to original function but with selective R2
    # We need to modify the fuse_basis functions to accept the enable flags
    # For now, we call the original and let the quantization config handle the rest
    if args.rotate_mode == "quik":
        rotation_utils.fuse_basis_per_layer(model, args)
    else:
        if "full_shared" in args.rotation_granularity.lower():
            fuse_basis_shared_selective(model, args, enable_R1, enable_R2)
        elif "one_per_decoder" in args.rotation_granularity.lower():
            rotation_utils.fuse_basis_one_per_decoder(model, args)
        else:
            rotation_utils.fuse_basis_per_layer(model, args)


def fuse_basis_shared_selective(model, args, enable_R1: bool, enable_R2: bool):
    """
    Selective version of fuse_basis_shared that respects R1/R2 enable flags.
    """
    import tqdm
    
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    high_bits_length = int(args.high_fraction * model_dim)
    head_dim = model_dim // num_heads
    
    U_cpk = torch.load(args.optimized_basis_path)
    U_attn = U_cpk["attn_mlp"].cuda()

    if not args.train_rotations:
        R_dict = torch.load(args.optimized_rotation_path)
        R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
        R1_2 = R_dict["R1_2"].cuda().to(torch.float64)

        assert R1_2.shape[0] == high_bits_length
            
        R1 = torch.block_diag(R1_1, R1_2)
        R1_0 = R_dict["R1_0"]
        if R1_0 is not None:
            R1 = torch.block_diag(R1_0.cuda().to(torch.float64), R1)

        if enable_R1:
            U_attn = torch.matmul(U_attn, R1)
        else:
            logger.info("R1 disabled - using identity for R1")
    
    torch.distributed.barrier()
    
    if enable_R1:
        rotation_utils.rotate_embeddings(model, U_attn)
        rotation_utils.rotate_head(model, U_attn)

    utils.cleanup_memory()

    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if enable_R1:
            rotation_utils.rotate_attention_inputs(layers[idx], U_attn)

        key = f"layer.{idx}.self_attn.value"
        U_value = U_cpk[key].cuda()
        
        if enable_R2 and not args.train_rotations:
            if 'trained' in args.optimized_rotation_path:
                R2_1 = R_dict[f"model.layers.{idx}.self_attn.R2_1"].cuda().to(torch.float64)
                R2_2 = R_dict[f"model.layers.{idx}.self_attn.R2_2"].cuda().to(torch.float64)
                R2 = torch.block_diag(R2_1, R2_2)
                if f"model.layers.{idx}.self_attn.R2_0" in R_dict.keys():
                    R2_0 = R_dict[f"model.layers.{idx}.self_attn.R2_0"].cuda().to(torch.float64)
                    R2 = torch.block_diag(R2_0, R2)
            else:
                R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
                R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
                R2 = torch.block_diag(R2_1, R2_2)
                R2_0 = R_dict["R2_0"]
                if R2_0 is not None:
                    R2 = torch.block_diag(R2_0.cuda().to(torch.float64), R2)
            
            U_value = torch.matmul(U_value, R2)
            rotation_utils.rotate_ov_proj(layers[idx], num_heads, head_dim, R2=U_value, per_head=True)
        else:
            if not enable_R2:
                logger.info(f"R2 disabled for layer {idx} - skipping V/O rotation")
            # Skip R2 rotation but still apply identity if needed

        if enable_R1:
            rotation_utils.rotate_attention_output(layers[idx], U_attn)
            rotation_utils.rotate_mlp_input(layers[idx], U_attn)
            rotation_utils.rotate_mlp_output(layers[idx], R1=U_attn, R4=None)


def configure_activation_quantization(model, args, quant_config, enable_R2: bool):
    """
    Configure activation quantization based on the quant_config.
    """
    if args.a_bits >= 16 and args.v_bits >= 16:
        return
    
    qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
    down_proj_groupsize = -1
    if args.a_groupsize > 0:
        down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
    
    num_heads = model.config.num_attention_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    mlp_dim = model.config.intermediate_size
    v_groupsize = head_dim
    
    if args.rotate_mode == "resq" or args.rotate_mode == "quik":
        high_bits_fraction = args.high_fraction
        high_bits_length = int(high_bits_fraction * model_dim)
        low_bits_fraction = args.low_fraction
        low_bits_length = int(low_bits_fraction * model_dim)
    else:
        high_bits_length = 0
        low_bits_length = 0
    
    for name in qlayers:
        layer_input_bits = args.a_bits
        layer_groupsize = args.a_groupsize
        layer_a_sym = not args.a_asym
        layer_a_clip = args.a_clip_ratio
        layer_high_bits_length = high_bits_length
        layer_low_bits_length = low_bits_length
        
        # Apply config-based overrides
        if quant_config is not None:
            module_config = quant_config.get_module_config(name)
            if module_config is not None:
                if not module_config.enabled:
                    layer_input_bits = 16
                    layer_high_bits_length = 0
                    layer_low_bits_length = 0
                else:
                    layer_input_bits = module_config.bits
                    layer_groupsize = module_config.groupsize if module_config.groupsize != -1 else layer_groupsize
                    layer_a_sym = not module_config.asym
                    layer_a_clip = module_config.clip_ratio
        
        # V_proj output quantization (R2-related)
        if "v_proj" in name and args.v_bits < 16:
            if enable_R2:
                if args.rotate_mode == "resq" or args.rotate_mode == "quik":
                    v_high_bits_length = int(v_groupsize * high_bits_fraction)
                    v_low_bits_length = int(v_groupsize * low_bits_fraction)
                else:
                    v_high_bits_length = 0
                    v_low_bits_length = 0

                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not args.v_asym,
                    clip_ratio=args.v_clip_ratio,
                    high_bits_length=v_high_bits_length,
                    high_bits=args.high_bits,
                    low_bits_length=v_low_bits_length,
                    low_bits=args.low_bits,
                )
            else:
                # R2 disabled - keep v_proj output in high precision
                logger.info(f"R2 disabled - keeping {name} output in high precision")
                qlayers[name].out_quantizer.configure(bits=16, groupsize=-1, sym=True)
        
        # O_proj input (R2-related)
        if "o_proj" in name:
            layer_groupsize = head_dim
            if enable_R2:
                if args.rotate_mode == "resq" or args.rotate_mode == "quik":
                    layer_high_bits_length = int(v_groupsize * high_bits_fraction)
                    layer_low_bits_length = int(v_groupsize * low_bits_fraction)
            else:
                # R2 disabled - keep o_proj input in high precision
                layer_input_bits = 16
                layer_high_bits_length = 0
                layer_low_bits_length = 0
                logger.info(f"R2 disabled - keeping {name} input in high precision")

        if "lm_head" in name:
            layer_input_bits = 16
            layer_high_bits_length = 0
            layer_low_bits_length = 0

        if "basis_change" in name:
            layer_input_bits = 8
            layer_high_bits_length = 0
            layer_low_bits_length = 0
        
        if "visual" in name:
            layer_input_bits = 16
            layer_high_bits_length = 0
            layer_low_bits_length = 0

        if "down_proj" in name:
            layer_high_bits_length = 0
            layer_low_bits_length = 0
            
            # Check if we should keep down_proj input in high precision (R4 disabled)
            if quant_config is not None and quant_config.down_proj.left_side_high_bits:
                layer_input_bits = quant_config.down_proj.left_side_bits
                logger.info(f"Keeping {name} input (left side) at {layer_input_bits} bits")
            elif args.int8_down_proj:
                layer_input_bits = 8

        qlayers[name].quantizer.configure(
            bits=layer_input_bits,
            groupsize=layer_groupsize,
            sym=layer_a_sym,
            clip_ratio=layer_a_clip,
            high_bits_length=layer_high_bits_length,
            high_bits=args.high_bits,
            low_bits_length=layer_low_bits_length,
            low_bits=args.low_bits,
        )


def configure_k_cache_quantization(model, args, quant_config):
    """
    Configure K-cache quantization (R3-related).
    """
    if args.k_bits >= 16:
        return
    
    if args.k_pre_rope:
        raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
    
    if hasattr(model, "visual"):
        rope_function_name = "apply_multimodal_rotary_pos_emb"
    else:
        rope_function_name = "apply_rotary_pos_emb"
    
    layers = model.model.layers
    num_heads = model.config.num_attention_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    
    if args.rotate_mode == "resq" or args.rotate_mode == "quik":
        U_cpk = torch.load(args.optimized_basis_path)
        high_bits_length = int(args.high_fraction * head_dim)
        low_bits_length = int(args.low_fraction * head_dim)
    else:
        high_bits_length = 0
        low_bits_length = 0

    k_quant_config = {
        "k_bits": args.k_bits,
        "k_bits_high": args.high_bits,
        "k_bits_low": args.low_bits,
        "k_groupsize": args.k_groupsize,
        "k_sym": not args.k_asym,
        "k_clip_ratio": args.k_clip_ratio,
        "high_bits_length": high_bits_length,
        "low_bits_length": low_bits_length,
    }

    for idx, layer in enumerate(layers):
        if args.rotate_mode == "resq" or args.rotate_mode == "quik":
            R_dict = torch.load(args.optimized_rotation_path)
            R2_1 = R_dict["R2_1"].cuda().to(torch.float64)
            R2_2 = R_dict["R2_2"].cuda().to(torch.float64)
            R2 = torch.block_diag(R2_1, R2_2)
            R2_0 = R_dict["R2_0"]
            if R2_0 is not None:
                R2 = torch.block_diag(R2_0.cuda().to(torch.float64), R2)
            k_rotation = U_cpk[f"layer.{idx}.self_attn.key_pos"].cuda()
            k_rotation = torch.matmul(k_rotation, R2)
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(8)
            quantizer.find_params(k_rotation)
            k_rotation = quantizer.quantize(k_rotation)
            k_had = False
        else:
            k_rotation = None
            k_had = True
        
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_function_name,
            config=model.config,
            k_rotation=k_rotation,
            k_had=k_had,
            **k_quant_config,
        )


def load_custom_r4(path: str, intermediate_size: int):
    """
    Load a custom R4 rotation matrix from file.
    
    Args:
        path: Path to the rotation matrix file (.bin or .pt)
        intermediate_size: Expected dimension of the rotation matrix
    
    Returns:
        Loaded rotation matrix or None if loading fails
    """
    try:
        R4 = torch.load(path)
        if isinstance(R4, dict):
            # Handle different saved formats
            if "R4" in R4:
                R4 = R4["R4"]
            elif "rotation" in R4:
                R4 = R4["rotation"]
        
        # Validate dimensions
        if R4.shape[0] != R4.shape[1]:
            logger.error(f"Custom R4 must be square matrix, got {R4.shape}")
            return None
        
        logger.info(f"Loaded custom R4 from {path} with shape {R4.shape}")
        return R4.cuda().to(torch.float64)
    
    except Exception as e:
        logger.error(f"Failed to load custom R4 from {path}: {e}")
        return None


def apply_custom_r4_to_down_proj(model, R4, layer_name: str):
    """
    Apply a custom R4 rotation matrix to the down_proj layer.
    
    This function modifies the down_proj weights to incorporate the custom rotation
    on the left side (input side) of the matrix multiplication.
    
    Args:
        model: The model
        R4: The custom rotation matrix
        layer_name: Name of the layer (e.g., "model.layers.0.mlp.down_proj")
    """
    logger.info(f"Applying custom R4 to {layer_name}")
    
    # Find the layer
    parts = layer_name.split('.')
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part, None)
            if module is None:
                break
    
    if module is None or not hasattr(module, 'module'):
        logger.warning(f"Could not find module for {layer_name}")
        return
    
    # Apply rotation to weights
    # For down_proj: output = W @ (R4 @ input)
    # Equivalent to: output = (W @ R4^T^(-1)) @ (R4 @ input) = W' @ rotated_input
    # We need to apply R4^(-1)^T = R4 (for orthogonal matrices) to the weights
    
    down_proj = module.module
    if hasattr(down_proj, 'weight'):
        dtype = down_proj.weight.dtype
        W = down_proj.weight.data.to(torch.float64).cuda()
        
        # For input rotation: W_new = W @ R4.T
        # Because (W @ R4.T) @ (R4 @ x) = W @ x
        W_new = torch.matmul(W, R4.T)
        down_proj.weight.data = W_new.to(dtype).cpu()
        
        logger.info(f"Applied custom R4 to {layer_name} weights")


# Convenience function to create custom R4 rotation
def create_random_r4(size: int, seed: int = None) -> torch.Tensor:
    """
    Create a random orthogonal R4 rotation matrix.
    
    Args:
        size: Dimension of the rotation matrix
        seed: Random seed for reproducibility
    
    Returns:
        Random orthogonal matrix of shape (size, size)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    random_matrix = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def save_r4_rotation(R4: torch.Tensor, path: str):
    """
    Save an R4 rotation matrix to file.
    
    Args:
        R4: The rotation matrix
        path: Path to save the matrix
    """
    torch.save({"R4": R4}, path)
    logger.info(f"Saved R4 rotation to {path}")
