#!/bin/bash
# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### Example: Using custom R4 rotation for down_proj
### Allows custom rotation matrix for the left side of down_proj

# First, create a custom R4 rotation using the helper script:
# python create_custom_r4.py --size 14336 --output ./rotation/custom_R4.bin

torchrun --nnodes=1 --nproc_per_node=1 --master_port=24544 ptq.py \
--input_model meta-llama/Llama-3.2-1B \
--per_device_eval_batch_size 1 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 16 \
--a_bits 4 \
--k_bits 4 \
--v_bits 4 \
--high_bits 8 \
--low_bits 2 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 64 \
--v_groupsize 64 \
--high_fraction 0.125 \
--low_fraction 0.0 \
--rotate_mode "resq" \
--optimized_rotation_path ./rotation/R-high-0.125-low-0.0-sparse-0.0-Llama-3.2-1B.bin \
--optimized_basis_path ./rotation/U-wikitext-512-Llama-3.2-1B.bin \
--rotation_granularity 'full_shared' \
--rotate \
--custom_r4_path ./rotation/custom_R4.bin \
--tasks "mmlu,boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"