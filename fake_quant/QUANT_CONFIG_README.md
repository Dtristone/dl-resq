# RESQ Configurable Rotation and Quantization

This document describes the new configurable rotation and quantization features added to the RESQ PTQ implementation.

## Overview

RESQ uses multiple rotation matrices to transform activations for efficient quantization:

- **R1**: Applied to hidden states (embeddings, q/k/v_proj inputs, o_proj output, up/gate_proj inputs, down_proj output)
- **R2**: Applied to V attention output and O_proj input (per-head rotation in attention computation)
- **R3**: Applied to K cache (key rotation after RoPE)
- **R4**: Applied to down_proj input (left side, online Hadamard transform)

## New Features

### 1. Selective Rotation/Quantization Control

You can now control which rotation matrices are applied and which modules are quantized using:

#### Command-line flags:
```bash
--use_r1_only_config     # Only rotate/quantize R1-related modules
--disable_R2_rotation    # Disable R2 (V/O attention) rotation
--disable_R3_rotation    # Disable R3 (K cache) rotation  
--disable_R4_rotation    # Disable R4 (down_proj Hadamard) rotation
```

#### Using R1-only mode:
```bash
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model meta-llama/Llama-3.2-1B \
    --rotate_mode "resq" \
    --use_r1_only_config \
    ... other args
```

### 2. Custom Quantization Config File

Create a JSON config file for fine-grained control:

```bash
--quant_config_path ./config/quant_config.json
```

Example config (`config/quant_config_r1_only.json`):
```json
{
    "rotation": {
        "enable_R1": true,
        "enable_R2": false,
        "enable_R3": false,
        "enable_R4": false,
        "custom_R4_path": null
    },
    "quantization": {
        "q_proj": {"enabled": true, "bits": 4, "asym": true},
        "k_proj": {"enabled": true, "bits": 4, "asym": true},
        "v_proj": {"enabled": true, "bits": 4, "output_quant_enabled": false},
        "o_proj": {"enabled": false, "bits": 16},
        "up_proj": {"enabled": true, "bits": 4},
        "gate_proj": {"enabled": true, "bits": 4},
        "down_proj": {"enabled": false, "bits": 16, "left_side_high_bits": true},
        "k_cache": {"enabled": false, "bits": 16},
        "v_cache": {"enabled": false, "bits": 16}
    },
    "mixed_precision": {
        "high_bits": 8,
        "low_bits": 2,
        "high_fraction": 0.125,
        "low_fraction": 0.0
    }
}
```

### 3. Custom R4 Rotation Matrix

Apply a custom rotation matrix to the down_proj left side (R4):

```bash
# Create custom R4 rotation
python create_custom_r4.py --size 14336 --output ./rotation/custom_R4.bin --type random

# Use in PTQ
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
    --input_model meta-llama/Llama-3.2-1B \
    --rotate_mode "resq" \
    --custom_r4_path ./rotation/custom_R4.bin \
    ... other args
```

R4 types available:
- `random`: Random orthogonal matrix (QR decomposition)
- `hadamard`: Hadamard matrix (requires power-of-2 size)
- `identity`: Identity matrix (no rotation)

## Module Configuration Options

Each module in the config supports:

| Option | Description |
|--------|-------------|
| `enabled` | Whether to quantize this module's input |
| `bits` | Quantization bits (4, 8, 16) |
| `groupsize` | Group size for quantization (-1 for per-token) |
| `asym` | Asymmetric quantization (true/false) |
| `clip_ratio` | Clipping ratio for quantization |
| `output_quant_enabled` | Quantize output (for v_proj) |
| `output_bits` | Output quantization bits |
| `left_side_high_bits` | Keep left side in high precision (for down_proj) |
| `left_side_bits` | Bits for left side (for down_proj) |

## R1-Only Configuration

When using `--use_r1_only_config`, the following modules are quantized:
- ✅ q_proj input (4 bits)
- ✅ k_proj input (4 bits)
- ✅ v_proj input (4 bits)
- ✅ up_proj input (4 bits)
- ✅ gate_proj input (4 bits)

And these are kept in high precision (16 bits):
- ❌ v_proj output (R2-related)
- ❌ o_proj input (R2-related)
- ❌ k_cache (R3-related)
- ❌ v_cache (R2-related)
- ❌ down_proj input/left side (R4-related)

## File Structure

```
fake_quant/
├── config/
│   ├── quant_config.json           # Default config (all rotations enabled)
│   └── quant_config_r1_only.json   # R1-only config
├── eval_utils/
│   ├── main.py                     # Updated to support config
│   └── ptq_configurable.py         # New configurable PTQ implementation
├── utils/
│   ├── process_args.py             # Updated with new arguments
│   └── quant_config_utils.py       # Config loading/parsing utilities
├── create_custom_r4.py             # Helper to create custom R4 rotation
├── 3_eval_ptq_r1_only.sh          # Example: R1-only evaluation
├── 3_eval_ptq_custom_config.sh    # Example: Custom config evaluation
└── 3_eval_ptq_custom_r4.sh        # Example: Custom R4 evaluation
```

## Python API

```python
from utils.quant_config_utils import (
    QuantizationConfig,
    load_quant_config,
    create_r1_only_config
)

# Load from file
config = load_quant_config("./config/quant_config.json")

# Create R1-only config programmatically
config = create_r1_only_config()

# Access module settings
print(config.q_proj.enabled)  # True
print(config.q_proj.bits)     # 4
print(config.rotation.enable_R1)  # True

# Modify settings
config.rotation.enable_R2 = True
config.down_proj.left_side_bits = 8

# Save to file
config.save_to_json("./config/my_config.json")
```

## Custom R4 Rotation API

```python
from eval_utils.ptq_configurable import (
    create_random_r4,
    save_r4_rotation,
    load_custom_r4
)

# Create and save R4
R4 = create_random_r4(size=14336, seed=42)
save_r4_rotation(R4, "./rotation/custom_R4.bin")

# Load R4
R4 = load_custom_r4("./rotation/custom_R4.bin", intermediate_size=14336)
```

## Notes

1. When using R1-only mode, the K and V caches are not rotated/quantized, which may increase memory usage but improves accuracy.

2. Custom R4 rotation is applied by modifying the down_proj weights at initialization time. The rotation is incorporated into the weight matrix.

3. The config file format is JSON for easy editing and version control.

4. Command-line flags (like `--disable_R2_rotation`) override config file settings.
