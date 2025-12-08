# ResQ Post-Training Quantization (PTQ) Flow Documentation

## Overview
This document provides a comprehensive explanation of the Post-Training Quantization (PTQ) flow in the ResQ project, specifically focusing on the `fake_quant/ptq.py` entry point and the detailed quantization process.

## Paper Reference
ResQ is based on the paper: "ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals" (https://arxiv.org/abs/2412.14363)

## High-Level PTQ Pipeline

The PTQ process in ResQ consists of two main phases:

### Phase 1: Get Basis (Preprocessing)
**Script**: `0_get_basis.sh` → `get_basis.py`

This phase computes the optimal rotation matrices and basis vectors for the quantization process:

1. **Load Model**: Load the full-precision model
2. **Fuse Layer Norms**: Combine layer normalization into adjacent linear layers
3. **Collect Activations**: Run calibration data through the model to collect activation statistics
4. **PCA Analysis**: Perform Principal Component Analysis to identify the low-rank subspace with highest variance
5. **Compute Basis Matrices**: 
   - Compute `U` matrices (PCA basis) for different layers
   - Save to `rotation/U-{dataset}-{nsamples}-{model}.bin`
6. **Optimize Rotations**:
   - Compute optimal rotation matrices `R` to suppress outliers
   - Save to `rotation/R-high-{high_fraction}-low-{low_fraction}-sparse-{sparse_fraction}-{model}.bin`

### Phase 2: Quantize and Evaluate
**Script**: `2_eval_ptq.sh` → `ptq.py`

This is the main quantization and evaluation phase.

---

## Detailed PTQ Flow in `ptq.py`

### Entry Point: `train()` function (Line 291)

```
1. Initialize distributed training environment
2. Parse arguments (model config, training config, PTQ config)
3. Set random seeds for reproducibility
4. Load model configuration
5. Apply flash attention if enabled
6. Handle word embeddings (clone lm_head if tied)
7. Load the appropriate model architecture (Llama/Qwen2/Qwen2-VL)
8. Initialize basis_change matrices to identity
9. Call ptq_model() to apply quantization → CORE FUNCTION
10. Load tokenizer
11. Evaluate quantized model
```

---

## Core PTQ Function: `ptq_model()` (eval_utils/main.py)

This is the heart of the quantization process. Here's the detailed flow:

### Step 1: Model Rotation and Basis Transformation (Lines 24-33)

**Purpose**: Transform the model to a space where quantization errors are minimized.

```python
if rotate_mode == "resq" or "quik":
    # 1.1 Fuse layer normalization into adjacent layers
    fuse_layer_norms(model)
    
    # 1.2 Apply the pre-computed PCA basis (U matrices) to the model
    rotation_utils.fuse_basis_to_model(model, args)
    #    - Loads U matrices from optimized_basis_path
    #    - Transforms weights: W' = W @ U
    #    - This rotates activations into PCA space
    
    # 1.3 Rearrange columns to group high/low variance dimensions
    rotation_utils.rearrange_columns(model, args, False)
    #    - Reorders dimensions: [low_bits_dims | mid_bits_dims | high_bits_dims]
    #    - Enables mixed-precision quantization
```

**Key Insight**: The rotation `U @ R` transforms activations such that:
- High-variance dimensions (outliers) are isolated
- Low-variance dimensions can be quantized more aggressively
- Minimizes overall quantization error

### Step 2: Add Activation Quantization Wrappers (Lines 35-43)

```python
# 2.1 Wrap all linear layers with ActQuantWrapper
quant_utils.add_actquant(model)

# 2.2 Configure special handling for down_proj layers
qlayers = quant_utils.find_qlayers(model)
for name in qlayers:
    if "down_proj" in name:
        # Apply Hadamard transform for down_proj
        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        qlayers[name].online_full_had = True
        qlayers[name].had_K = had_K
        qlayers[name].K = K
```

**Purpose**: Prepare layers for activation quantization by wrapping them with quantization-aware modules.

### Step 3: Weight Quantization (Lines 49-73)

**Three options for weight quantization**:

#### Option A: Load Pre-quantized Model (Lines 51-58)
```python
if args.load_qmodel_path:
    save_dict = torch.load(args.load_qmodel_path)
    model.load_state_dict(save_dict["model"])
```

#### Option B: GPTQ Quantization (Lines 60-69)
```python
elif not args.w_rtn:
    # Load calibration data (WikiText-2)
    trainloader = data_utils.get_wikitext2(...)
    
    # Apply GPTQ (Gradient-based PTQ with Quantization)
    quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
    #    - For each layer:
    #      1. Collect Hessian information from calibration data
    #      2. Use second-order information to minimize quantization error
    #      3. Quantize weights with optimal scales
```

**GPTQ Algorithm** (in `gptq_utils.py`):
- Iterates through layers sequentially
- For each weight matrix:
  1. Computes Hessian: H = 2 * X^T @ X / nsamples
  2. Uses block-wise quantization with Cholesky decomposition
  3. Applies mixed-precision: different bit-widths for different dimensions
  4. Updates remaining weights to compensate for quantization error

#### Option C: RTN (Round-To-Nearest) (Lines 71-73)
```python
else:
    # Simple rounding without optimization
    quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
```

### Step 4: Activation Quantization Configuration (Lines 80-166)

**Purpose**: Configure per-layer activation quantization with mixed precision.

```python
# 4.1 Calculate mixed-precision dimensions
high_bits_fraction = args.high_fraction  # e.g., 0.125 (1/8)
high_bits_length = int(high_bits_fraction * model_dim)
low_bits_fraction = args.low_fraction    # e.g., 0.0
low_bits_length = int(low_bits_fraction * model_dim)

# 4.2 Configure each layer
for name in qlayers:
    if "v_proj" in name:
        # Value projection: per-group mixed precision
        v_high_bits_length = int(v_groupsize * high_bits_fraction)
        qlayers[name].out_quantizer.configure(
            bits=args.v_bits,              # e.g., 4-bit
            high_bits=args.high_bits,      # e.g., 8-bit for high variance dims
            low_bits=args.low_bits,        # e.g., 2-bit for low variance dims
            high_bits_length=v_high_bits_length,
            ...
        )
    
    elif "o_proj" in name:
        # Output projection: similar mixed precision
        
    elif "basis_change" in name:
        # Basis transformation: always 8-bit
        layer_input_bits = 8
        
    elif "down_proj" in name:
        # Down projection: optional 8-bit or 4-bit
        if args.int8_down_proj:
            layer_input_bits = 8
    
    # Configure the quantizer
    qlayers[name].quantizer.configure(
        bits=layer_input_bits,
        groupsize=layer_groupsize,
        sym=layer_a_sym,
        clip_ratio=layer_a_clip,
        high_bits_length=high_bits_length,
        high_bits=args.high_bits,
        low_bits_length=low_bits_length,
        low_bits=args.low_bits,
    )
```

**Mixed-Precision Layout**:
```
Activation vector: [low_bits_dims | mid_bits_dims | high_bits_dims]
                    [  2-bit     |    4-bit      |    8-bit     ]
                    [   0%       |    87.5%      |    12.5%     ]
```

### Step 5: KV Cache Quantization (Lines 168-225)

**Purpose**: Quantize the Key cache in attention mechanism.

```python
if args.k_bits < 16:
    # 5.1 Load pre-computed rotation matrices
    if args.rotate_mode == "resq":
        U_cpk = torch.load(args.optimized_basis_path)
        R_dict = torch.load(args.optimized_rotation_path)
        
        # Combine rotations: U @ R
        R2 = torch.block_diag(R2_1, R2_2)
        k_rotation = U_cpk[f"layer.{idx}.self_attn.key_pos"]
        k_rotation = torch.matmul(k_rotation, R2)
        
        # 5.2 Quantize the rotation matrix itself to 8-bit
        quantizer = quant_utils.WeightQuantizer()
        quantizer.configure(8)
        k_rotation = quantizer.quantize(k_rotation)
    
    # 5.3 Add rotation wrapper after RoPE
    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
        layer.self_attn,
        rope_function_name,
        k_rotation=k_rotation,
        k_bits=args.k_bits,
        high_bits_length=high_bits_length,
        ...
    )
```

**Key Cache Quantization Flow**:
1. Keys are computed: K = X @ W_k
2. RoPE (Rotary Position Embedding) is applied
3. **Rotation is applied**: K' = K @ R (suppresses outliers)
4. **Quantization**: K_quant = Quantize(K', mixed_precision)
5. K_quant is stored in KV cache

---

## Quantization Details

### Mixed-Precision Quantization Strategy

ResQ uses a **provably optimal** mixed-precision strategy:

```
Given:
- Total bit budget B
- Activation variances σ²₁, σ²₂, ..., σ²ₙ

Optimal allocation:
- Allocate more bits to high-variance dimensions
- Minimize reconstruction error: E = Σ σ²ᵢ / 2^(2bᵢ)

Implementation:
1. Use PCA to find dimensions sorted by variance
2. Keep top 12.5% in 8-bit (high_fraction=0.125)
3. Quantize remaining 87.5% to 4-bit
4. Optional: keep bottom dims in 2-bit
```

### Quantization Parameters

**Typical Configuration** (from `2_eval_ptq.sh`):
```
--w_bits 16          # Weight bits (or 4 with GPTQ)
--a_bits 4           # Activation bits (mid precision)
--k_bits 4           # Key cache bits
--v_bits 4           # Value cache bits
--high_bits 8        # High precision bits
--low_bits 2         # Low precision bits
--high_fraction 0.125  # Fraction in high precision
--low_fraction 0.0     # Fraction in low precision
--k_groupsize 128    # Group size for per-group quantization
```

### Quantizer Implementation

**ActQuantizer** (in `quant_utils.py`):

```python
def forward(self, x):
    # Split activation into three parts
    low_dim = self.low_bits_length
    high_dim = x.shape[-1] - self.high_bits_length
    
    x_l = x[..., :low_dim]          # Low precision part
    x_m = x[..., low_dim:high_dim]  # Mid precision part
    x_h = x[..., high_dim:]         # High precision part
    
    # Quantize each part with different precision
    x_m = STEQuantize.apply(x_m, self.scale, self.maxq)        # 4-bit
    x_h = STEQuantize.apply(x_h, self.scale_h, self.maxq_h)    # 8-bit
    x_l = STEQuantize.apply(x_l, self.scale_l, self.maxq_l)    # 2-bit
    
    # Concatenate back
    return torch.cat([x_l, x_m, x_h], dim=-1)
```

**Scale Calculation**:
- **Symmetric**: `scale = max(abs(x)) / (2^(bits-1) - 1)`
- **Asymmetric**: `scale = (max(x) - min(x)) / (2^bits - 1)`
- Per-token or per-group granularity

---

## Evaluation Flow (evaluate() function in ptq.py)

After quantization, the model is evaluated:

```
1. Perplexity Evaluation on WikiText-2
   - Uses eval_utils.evaluator()
   - Measures how well the quantized model predicts text
   
2. Task Evaluation (via lm-eval-harness)
   - Tasks: MMLU, BoolQ, PIQA, HellaSwag, etc.
   - Uses HFLM wrapper for standard tasks
   - Uses HFMultimodalLM for vision tasks
   
3. LongBench Evaluation (optional)
   - Long-context tasks (narrativeqa, qasper, etc.)
   - Custom evaluation with specific prompts
```

---

## Key Files and Their Roles

```
fake_quant/
├── ptq.py                          # Main entry point
├── get_basis.py                    # Compute PCA basis and rotations
├── eval_utils/
│   ├── main.py                     # ptq_model() core logic
│   ├── rotation_utils.py           # Rotation and basis transformation
│   └── gptq_utils.py              # GPTQ weight quantization
├── utils/
│   ├── quant_utils.py             # Quantizer implementations
│   ├── hadamard_utils.py          # Hadamard transform utilities
│   ├── fuse_norm_utils.py         # Layer norm fusion
│   └── data_utils.py              # Data loading
└── rotation/                       # Pre-computed matrices
    ├── U-*.bin                     # PCA basis matrices
    └── R-*.bin                     # Rotation matrices
```

---

## Complete Execution Flow

### Step-by-Step Execution

**Step 1**: Compute Basis (once per model)
```bash
bash 0_get_basis.sh
# Runs: get_basis.py
# Output: rotation/U-*.bin, rotation/R-*.bin
```

**Step 2**: Quantize and Evaluate
```bash
bash 2_eval_ptq.sh
# Runs: ptq.py with torchrun
```

**Detailed Flow in ptq.py**:
```
1. train()
   ↓
2. Load model (LlamaForCausalLM/Qwen2ForCausalLM)
   ↓
3. ptq_model(args, model, model_args)
   ↓
4. Rotation Phase:
   - fuse_layer_norms()
   - fuse_basis_to_model()      # Apply U
   - rearrange_columns()         # Sort by variance
   ↓
5. Weight Quantization:
   - gptq_fwrd() OR rtn_fwrd()
   ↓
6. Activation Quantization Setup:
   - add_actquant()
   - Configure each layer's quantizer
   ↓
7. KV Cache Quantization Setup:
   - Load and quantize rotation matrices
   - Add rotation wrappers after RoPE
   ↓
8. evaluate(model, tokenizer, args)
   ↓
9. Output results
```

---

## Advanced Features

### 1. Hadamard Transform for down_proj
- Applied online during forward pass
- Further suppresses outliers in MLP activations
- Computational overhead is minimal

### 2. Straight-Through Estimator (STE)
- Used for gradient-based fine-tuning (if needed)
- `forward`: quantize
- `backward`: pass gradient unchanged

### 3. Group-wise Quantization
- Divides activations into groups
- Computes separate scales per group
- Better handles local variations

### 4. Rotation Matrix Quantization
- Even the rotation matrices R are quantized to 8-bit
- Reduces memory overhead
- Minimal accuracy loss

---

## Performance Characteristics

**From the paper**:
- **Perplexity**: Up to 33% lower than SpinQuant on WikiText
- **Speedup**: 2.4x over 16-bit baseline
- **Precision**: W16A4K4V4 (16-bit weights, 4-bit activations/KV cache)
- **Mixed Precision**: 12.5% in 8-bit, 87.5% in 4-bit

**Memory Savings**:
- Activations: 4x reduction (16→4 bit)
- KV Cache: 4x reduction (16→4 bit)
- Total model size: ~4x smaller in deployment

---

## Configuration Examples

### Example 1: Full ResQ (W16A4K4V4)
```bash
--w_bits 16 --a_bits 4 --k_bits 4 --v_bits 4 \
--high_bits 8 --high_fraction 0.125 \
--rotate_mode "resq"
```

### Example 2: ResQ with 4-bit Weights
```bash
--w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
--high_bits 8 --high_fraction 0.125 \
--rotate_mode "resq" \
--w_rtn False  # Use GPTQ for weights
```

### Example 3: Baseline (no quantization)
```bash
--w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 \
--rotate_mode "none"
```

---

## Debugging and Visualization

To understand the quantization effects:

1. **Check rotation matrices**:
```python
U_cpk = torch.load("rotation/U-*.bin")
print(U_cpk.keys())  # See which layers have basis
```

2. **Inspect quantizer configuration**:
```python
for name, module in model.named_modules():
    if hasattr(module, 'quantizer'):
        print(f"{name}: {module.quantizer.bits} bits")
```

3. **Monitor quantization error**:
- Compare fp16 vs quantized outputs
- Check activation ranges before/after quantization

---

## Summary

The ResQ PTQ flow achieves state-of-the-art quantization through:

1. **Optimal Mixed Precision**: PCA-based allocation of bit budgets
2. **Rotation-based Outlier Suppression**: Random rotations minimize outliers
3. **Low-rank Residual**: High-precision subspace for critical information
4. **GPTQ Weight Quantization**: Second-order optimization for weights
5. **Efficient Implementation**: Quantized rotations, online Hadamard

This enables 4-bit quantization of LLMs with minimal accuracy degradation and significant speedup.
