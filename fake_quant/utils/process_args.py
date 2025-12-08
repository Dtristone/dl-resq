# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional, Tuple

import argparse
import transformers


@dataclass
class ModelArguments:
    input_model: Optional[str] = field(
        default="test-input", metadata={"help": "Input model"}
    )
    output_rotation_path: Optional[str] = field(
        default="test-output", metadata={"help": "Output rotation checkpoint path"}
    )
    optimized_rotation_path: Optional[str] = field(
        default=None, metadata={"help": "Optimized rotation checkpoint path"}
    )

    optimized_basis_path: Optional[str] = field(
        default=None, metadata={"help": "Optimized rotation checkpoint path"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )


def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )

    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    parser.add_argument(
        "--rotate_mode",
        type=str,
        default=None,
        choices=["resq", "quarot", "spinquant", "quik", "none"],
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,
        help="Random Seed for generating random matrix!!",
    )
    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",
    )
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument(
        "--high_bits",
        type=int,
        default=16,
        help="""Number of bits for high precision component when doing mixed precision quantization""",
    )
    parser.add_argument(
        "--low_bits",
        type=int,
        default=16,
        help="""Number of bits for high precision component when doing mixed precision quantization""",
    )

    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--high_fraction",
        type=float,
        default=0,
        help="Fraction of channels to keep in floating point.",
    )

    parser.add_argument(
        "--low_fraction",
        type=float,
        default=0,
        help="Fraction of channels to keep in floating point.",
    )

    parser.add_argument(
        "--sparse_fraction",
        type=float,
        default=0.0,
        help="Fraction of channels to prune.",
    )

    parser.add_argument(
        "--down_proj_blocksize",
        type=int,
        default=1024,
        help="Block size for mixing down proj layer channels",
    )

    # Save/Load Quantized Model Arguments
    parser.add_argument(
        "--load_qmodel_path",
        type=str,
        default=None,
        help="Load the quantized model from the specified path!",
    )
    parser.add_argument(
        "--save_qmodel_path",
        type=str,
        default=None,
        help="Save the quantized model to the specified path!",
    )

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext",
        help="Dataset to use for calibration",
    )
    parser.add_argument("--tasks", default="")
    parser.add_argument("--long_bench_tasks", default="")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    parser.add_argument("--flash_attn", action="store_true")
    parser.add_argument("--rotation_granularity", default="per_layer")

    parser.add_argument("--train_rotations", action="store_true")

    # Experiments Arguments
    parser.add_argument(
        "--capture_layer_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the input and output of the specified decoder layer and dump into a file",
    )
    parser.add_argument(
        "--layerwise_mse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="calculates layerwise mean squared error of quantized input and full precision input",
    )

    parser.add_argument(
        "--layerwise_kurt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="calculates layerwise kurtosis of activations",
    )
    parser.add_argument(
        "--layerwise_shapiro",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="calculates layerwise shapriro gausianity test probabilities of activations",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=10, help="Which decoder layer to capture"
    )

    parser.add_argument(
        "--multigpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model parallelism using this flag. Map decoder blocks to different GPUs",
    )
    
    # Quantization Config Arguments
    parser.add_argument(
        "--quant_config_path",
        type=str,
        default=None,
        help="Path to the JSON quantization config file for selective rotation/quantization control",
    )
    parser.add_argument(
        "--use_r1_only_config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use R1-only config: only rotate and quantize R1-related modules (q/k/v_proj inputs, up/gate_proj, down_proj output)",
    )
    parser.add_argument(
        "--custom_r4_path",
        type=str,
        default=None,
        help="Path to custom R4 rotation matrix for down_proj left side",
    )
    parser.add_argument(
        "--disable_R2_rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable R2 rotation (V/O attention computation rotation)",
    )
    parser.add_argument(
        "--disable_R3_rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable R3 rotation (K cache rotation)",
    )
    parser.add_argument(
        "--disable_R4_rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable R4 rotation (down_proj left side Hadamard)",
    )
    
    parser.add_argument(
        "--use_mxfp4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use MXFP4 (Microscaling FP4) quantization instead of INT quantization",
    )
    parser.add_argument(
        "--mxfp4_group_size",
        type=int,
        default=32,
        help="Group size for MXFP4 quantization (default: 32)",
    )
    parser.add_argument(
        "--mxfp4_stochastic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use stochastic rounding for MXFP4 quantization",
    )
    parser.add_argument(
        "--mxfp4_targets",
        type=str,
        default="all",
        help="Comma-separated list of modules to apply MXFP4 (e.g., 'q_proj,k_proj,v_proj' or 'all')",
    )

    args, unknown = parser.parse_known_args()

    assert (
        args.a_groupsize == args.w_groupsize
    ), "a_groupsize should be the same as w_groupsize!"
    assert args.k_pre_rope is False, "Pre-RoPE quantization is not supported yet!"

    return args, unknown


def process_args_ptq():
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)
    if model_args.optimized_rotation_path is not None:
        ptq_args.optimized_rotation_path = model_args.optimized_rotation_path
    else:
        ptq_args.optimized_rotation_path = None

    if model_args.optimized_basis_path is not None:
        ptq_args.optimized_basis_path = model_args.optimized_basis_path
    else:
        ptq_args.optimized_basis_path = None

    ptq_args.bsz = training_args.per_device_eval_batch_size
    ptq_args.output_dir = training_args.output_dir

    # Load quantization config if provided
    from utils.quant_config_utils import load_quant_config, create_r1_only_config
    
    if ptq_args.use_r1_only_config:
        ptq_args.quant_config = create_r1_only_config()
    elif ptq_args.quant_config_path is not None:
        ptq_args.quant_config = load_quant_config(ptq_args.quant_config_path)
    else:
        ptq_args.quant_config = None
    
    # Apply command-line overrides for rotation control
    if ptq_args.quant_config is not None:
        if ptq_args.disable_R2_rotation:
            ptq_args.quant_config.rotation.enable_R2 = False
        if ptq_args.disable_R3_rotation:
            ptq_args.quant_config.rotation.enable_R3 = False
        if ptq_args.disable_R4_rotation:
            ptq_args.quant_config.rotation.enable_R4 = False
        if ptq_args.custom_r4_path is not None:
            ptq_args.quant_config.rotation.custom_R4_path = ptq_args.custom_r4_path

    return model_args, training_args, ptq_args
