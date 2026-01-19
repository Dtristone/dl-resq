#!/bin/bash
# MXFP4 Quantization Evaluation Script

torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model meta-llama/Llama-3.2-1B \
--output_dir ./outputs \
--do_eval \
--per_device_eval_batch_size 8 \
--model_max_length 2048 \
--fp16 \
--w_bits 4 \
--a_bits 4 \
--v_bits 4 \
--k_bits 4 \
--w_asym \
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
--use_mxfp4 \
--mxfp4_group_size 32 \
--mxfp4_targets "q_proj,k_proj,v_proj,up_proj,gate_proj" \
--tasks "mmlu,boolq,piqa"