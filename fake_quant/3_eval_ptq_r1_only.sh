# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### Example: R1-only quantization (only rotate and quantize R1-related modules)
### This keeps R2 (V/O attention), R3 (K cache), R4 (down_proj left side) in high precision

# Using the --use_r1_only_config flag:
torchrun --nnodes=1 --nproc_per_node=1 --master_port=24544 ptq.py \
--input_model meta-llama/Llama-3.2-1B \
--per_device_eval_batch_size 1 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 16 \
--a_bits 4 \
--k_bits 16 \
--v_bits 16 \
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
--use_r1_only_config \
--tasks "mmlu,boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
