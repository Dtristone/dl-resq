# ResQ PTQ Flow Diagram

## Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ResQ PTQ PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: Preprocessing (get_basis.py)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  1. Load Full-Precision Model                                          │
│     ↓                                                                   │
│  2. Fuse Layer Norms                                                    │
│     ↓                                                                   │
│  3. Collect Activations on Calibration Data (WikiText-2)               │
│     ↓                                                                   │
│  4. Perform PCA Analysis                                                │
│     • Compute covariance of activations                                │
│     • Find eigenvectors (principal components)                         │
│     ↓                                                                   │
│  5. Generate Basis Matrices (U)                                         │
│     • Save to: rotation/U-{dataset}-{nsamples}-{model}.bin            │
│     ↓                                                                   │
│  6. Optimize Rotation Matrices (R)                                      │
│     • Minimize outliers via random rotation                            │
│     • Save to: rotation/R-high-{h}-low-{l}-sparse-{s}-{model}.bin    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 2: Quantization & Evaluation (ptq.py)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ptq.py::train()                                                        │
│     ↓                                                                   │
│  1. Load Model & Config                                                 │
│     ↓                                                                   │
│  2. Call ptq_model() ──────────────────┐                               │
│                                        │                               │
└────────────────────────────────────────┼───────────────────────────────┘
                                         │
┌────────────────────────────────────────┼───────────────────────────────┐
│  eval_utils/main.py::ptq_model()       ↓                               │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  STEP 1: Rotation & Basis Transformation                         │ │
│  ├──────────────────────────────────────────────────────────────────┤ │
│  │  1.1 Fuse Layer Norms                                            │ │
│  │      • Combine LayerNorm into adjacent Linear layers             │ │
│  │      • W' = LayerNorm.weight * W / sqrt(variance)                │ │
│  │      ↓                                                            │ │
│  │  1.2 Apply PCA Basis (U matrices)                                │ │
│  │      • Load U from rotation/U-*.bin                              │ │
│  │      • Transform: W' = W @ U                                     │ │
│  │      • Rotates activations to PCA eigenspace                     │ │
│  │      ↓                                                            │ │
│  │  1.3 Rearrange Columns by Variance                               │ │
│  │      • Order: [low_var | mid_var | high_var]                     │ │
│  │      • Enables mixed-precision quantization                      │ │
│  │      ↓                                                            │ │
│  │  1.4 Add Activation Quantization Wrappers                        │ │
│  │      • Wrap all Linear layers with ActQuantWrapper               │ │
│  │      ↓                                                            │ │
│  │  1.5 Configure Hadamard for down_proj                            │ │
│  │      • Online Hadamard transform to spread outliers              │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  STEP 2: Weight Quantization                                     │ │
│  ├──────────────────────────────────────────────────────────────────┤ │
│  │  Choose one of:                                                  │ │
│  │                                                                   │ │
│  │  Option A: Load Pre-Quantized Model                              │ │
│  │     • Load from args.load_qmodel_path                            │ │
│  │                                                                   │ │
│  │  Option B: GPTQ (Recommended)                                    │ │
│  │     • Load calibration data (WikiText-2)                         │ │
│  │     • For each layer:                                            │ │
│  │       1. Compute Hessian H = 2 * X^T @ X / nsamples              │ │
│  │       2. Cholesky decomposition: H = L @ L^T                     │ │
│  │       3. Block-wise quantization with error compensation         │ │
│  │       4. Update remaining weights: W_new = W - error @ H^-1      │ │
│  │     • Mixed precision: different bits for different dims         │ │
│  │                                                                   │ │
│  │  Option C: RTN (Round-To-Nearest)                                │ │
│  │     • scale = max(|W|) / (2^(bits-1) - 1)                        │ │
│  │     • W_quant = round(W / scale) * scale                         │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  STEP 3: Activation Quantization Configuration                   │ │
│  ├──────────────────────────────────────────────────────────────────┤ │
│  │  Calculate Mixed-Precision Layout:                               │ │
│  │  ┌───────┬──────────┬──────────┐                                 │ │
│  │  │ Low   │   Mid    │  High    │                                 │ │
│  │  │ 2-bit │   4-bit  │  8-bit   │                                 │ │
│  │  │  0%   │  87.5%   │  12.5%   │  (typical: high_fraction=0.125)│ │
│  │  └───────┴──────────┴──────────┘                                 │ │
│  │                                                                   │ │
│  │  Configure each layer type:                                      │ │
│  │  • v_proj: Mixed precision per attention head group              │ │
│  │  • o_proj: Mixed precision per head dimension                    │ │
│  │  • down_proj: 8-bit (high outliers) or 4-bit                     │ │
│  │  • basis_change: Always 8-bit                                    │ │
│  │  • lm_head: Keep 16-bit (no quantization)                        │ │
│  │  • visual: Keep 16-bit (for VLMs)                                │ │
│  │                                                                   │ │
│  │  Quantizer Configuration:                                        │ │
│  │  • bits: base precision (e.g., 4)                                │ │
│  │  • groupsize: per-group quantization (-1 for per-token)          │ │
│  │  • sym: symmetric vs asymmetric quantization                     │ │
│  │  • clip_ratio: outlier clipping (default 1.0)                    │ │
│  │  • high_bits: precision for high-variance dims                   │ │
│  │  • low_bits: precision for low-variance dims                     │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                            ↓                                            │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  STEP 4: KV Cache Quantization                                   │ │
│  ├──────────────────────────────────────────────────────────────────┤ │
│  │  For each layer:                                                 │ │
│  │    1. Load PCA basis U for Key                                   │ │
│  │    2. Load rotation matrix R                                     │ │
│  │    3. Combine: k_rotation = U @ R                                │ │
│  │    4. Quantize k_rotation to 8-bit (save memory)                 │ │
│  │    5. Add rotation wrapper after RoPE:                           │ │
│  │                                                                   │ │
│  │       Forward pass:                                              │ │
│  │       K = X @ W_k                                                │ │
│  │       K = RoPE(K)          ← Apply rotary position embedding     │ │
│  │       K = K @ k_rotation   ← Apply PCA + rotation                │ │
│  │       K = Quantize(K)      ← Mixed-precision quantization        │ │
│  │       [Store K in cache]                                         │ │
│  │                                                                   │ │
│  │  Value cache is already handled via v_proj in Step 3             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────┼───────────────────────────────┐
│  Back to ptq.py::train()               ↓                               │
│                                                                         │
│  3. Load Tokenizer                                                      │
│     ↓                                                                   │
│  4. Call evaluate()                                                     │
│     ├─ Perplexity on WikiText-2                                        │
│     ├─ Downstream tasks (MMLU, BoolQ, etc.)                            │
│     └─ LongBench tasks (optional)                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Detailed Forward Pass with Quantization

### Standard Transformer Layer (No Quantization)

```
Input X (fp16)
    ↓
┌───────────────────────────────────────────┐
│ Attention Block                           │
├───────────────────────────────────────────┤
│ Q = X @ W_q                               │
│ K = X @ W_k                               │
│ V = X @ W_v                               │
│ Attn = softmax(Q @ K^T / √d)              │
│ Out = Attn @ V                            │
│ X = X + Out @ W_o                         │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ MLP Block                                 │
├───────────────────────────────────────────┤
│ Up = X @ W_up                             │
│ Gate = X @ W_gate                         │
│ Hidden = SiLU(Gate) * Up                  │
│ Out = Hidden @ W_down                     │
│ X = X + Out                               │
└───────────────────────────────────────────┘
    ↓
Output X (fp16)
```

### ResQ Quantized Transformer Layer

```
Input X (fp16)
    ↓ [Apply basis_change: X' = X @ U]
    ↓ [Quantize to 8-bit]
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ Attention Block (Quantized)                                           │
├───────────────────────────────────────────────────────────────────────┤
│ X_in = X                                                              │
│ X_q = Quantize_mixed(X_in)  [4-bit mid, 8-bit high, 2-bit low]       │
│                                                                       │
│ Q = X_q @ W_q (quantized)                                             │
│ K = X_q @ W_k (quantized)                                             │
│ V = X_q @ W_v (quantized)                                             │
│     ↓                                                                 │
│ Q = RoPE(Q)                                                           │
│ K = RoPE(K)                                                           │
│     ↓                                                                 │
│ K = K @ R (rotation)  ← Suppress outliers                            │
│     ↓                                                                 │
│ K_quant = Quantize_mixed(K)  [Store in KV cache]                     │
│ V_quant = Quantize_mixed(V)  [Store in KV cache]                     │
│     ↓                                                                 │
│ Attn = softmax(Q @ K_quant^T / √d)                                    │
│ Out = Attn @ V_quant                                                  │
│     ↓                                                                 │
│ Out_q = Quantize_mixed(Out)                                           │
│ O = Out_q @ W_o (quantized)                                           │
│ X = X + O                                                             │
└───────────────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────────────┐
│ MLP Block (Quantized)                                                 │
├───────────────────────────────────────────────────────────────────────┤
│ X_in = X                                                              │
│ X_q = Quantize_mixed(X_in)                                            │
│                                                                       │
│ Up = X_q @ W_up (quantized)                                           │
│ Gate = X_q @ W_gate (quantized)                                       │
│ Hidden = SiLU(Gate) * Up                                              │
│     ↓                                                                 │
│ Hidden = Hadamard(Hidden)  ← Spread outliers (for down_proj)         │
│     ↓                                                                 │
│ Hidden_q = Quantize(Hidden, 8-bit)  ← Higher precision for outliers  │
│ Out = Hidden_q @ W_down (quantized)                                   │
│ X = X + Out                                                           │
└───────────────────────────────────────────────────────────────────────┘
    ↓
Output X (fp16)
```

## Mixed-Precision Quantization Details

### Dimension Partitioning

```
Original Activation: [4096 dimensions, all in fp16]

After PCA Basis Transformation (U):
[4096 dimensions, sorted by variance]
high_var ← ← ← ← ← ← ← ← ← → → → → → → → → low_var

After Rearrangement:
┌────────────┬──────────────────────────────────┬────────────────┐
│  Low Var   │         Mid Var                  │   High Var     │
│   (0%)     │         (87.5%)                  │    (12.5%)     │
│  0 dims    │        3584 dims                 │    512 dims    │
│  (2-bit)   │         (4-bit)                  │    (8-bit)     │
└────────────┴──────────────────────────────────┴────────────────┘
    ↓              ↓                                   ↓
Skip or 2-bit   Main precision                  High precision
quantization    (most data)                     (outliers)

Total bits = 0*2 + 3584*4 + 512*8 = 14,336 + 4,096 = 18,432 bits
Average per dimension = 18,432 / 4,096 = 4.5 bits

Compare to uniform 4-bit: 4,096 * 4 = 16,384 bits (saves memory)
Compare to uniform 16-bit: 4,096 * 16 = 65,536 bits (4x reduction)
```

### Quantization Process

```
Input: X ∈ ℝ^(batch × seq × 4096)

Step 1: Split by precision
X_low  = X[..., :0]          # dims 0-0 (empty in this config)
X_mid  = X[..., 0:3584]      # dims 0-3583
X_high = X[..., 3584:4096]   # dims 3584-4095

Step 2: Compute scales per-token
For each token i:
  scale_mid[i]  = max(abs(X_mid[i]))  / (2^3 - 1) = max / 7  (4-bit sym)
  scale_high[i] = max(abs(X_high[i])) / (2^7 - 1) = max / 127 (8-bit sym)

Step 3: Quantize
X_mid_q  = round(X_mid  / scale_mid)  * scale_mid
X_high_q = round(X_high / scale_high) * scale_high

Step 4: Concatenate
X_quant = [X_mid_q || X_high_q]

Result: Mixed-precision quantized activation
```

## File Structure and Data Flow

```
fake_quant/
│
├── ptq.py                          ← Entry point
│   └─ train()
│       ├─ Load model
│       ├─ Call ptq_model() ────────┐
│       └─ Call evaluate()          │
│                                    │
├── eval_utils/                      │
│   ├── main.py                      ← Core quantization logic
│   │   └─ ptq_model() ◄────────────┘
│   │       ├─ Rotation & basis
│   │       ├─ Weight quantization (GPTQ/RTN)
│   │       ├─ Activation quantization
│   │       └─ KV cache quantization
│   │
│   ├── rotation_utils.py           ← Rotation transformations
│   │   ├─ fuse_basis_to_model()
│   │   ├─ rearrange_columns()
│   │   └─ add_qk_rotation_wrapper_after_function_call_in_forward()
│   │
│   └── gptq_utils.py               ← Weight quantization
│       ├─ gptq_fwrd()
│       └─ rtn_fwrd()
│
├── utils/
│   ├── quant_utils.py              ← Quantizer classes
│   │   ├─ ActQuantizer (activation quantization)
│   │   ├─ WeightQuantizer (weight quantization)
│   │   ├─ add_actquant() (wrap layers)
│   │   └─ find_qlayers() (find wrapped layers)
│   │
│   ├── fuse_norm_utils.py          ← Layer norm fusion
│   ├── hadamard_utils.py           ← Hadamard transforms
│   └── data_utils.py               ← Data loading
│
├── rotation/                        ← Pre-computed matrices
│   ├── U-wikitext-512-*.bin        ← PCA basis matrices
│   └── R-high-*-low-*-*.bin        ← Rotation matrices
│
└── get_basis.py                    ← Compute U and R matrices
    └─ get_outlier_rotations()
        ├─ Collect activations
        ├─ Compute PCA (U matrices)
        └─ Optimize rotations (R matrices)
```

## Key Algorithms

### GPTQ Weight Quantization (Simplified)

```python
# For each layer's weight matrix W:
def gptq(W, X_calibration, bits):
    """
    W: weight matrix (out_features, in_features)
    X_calibration: calibration activations (batch, in_features)
    bits: quantization bits
    """
    # 1. Compute Hessian from calibration data
    H = 2 * X_calibration.T @ X_calibration / batch_size
    
    # 2. Add damping for numerical stability
    diag = torch.diag(H)
    H += torch.diag(diag * percdamp)
    
    # 3. Cholesky decomposition
    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    
    # 4. Block-wise quantization
    for block_start in range(0, in_features, blocksize):
        block_end = block_start + blocksize
        
        for col in range(block_start, block_end):
            # Quantize column
            W_col = W[:, col]
            scale = max(abs(W_col)) / (2^(bits-1) - 1)
            W_col_q = round(W_col / scale) * scale
            
            # Compute error
            error = W_col - W_col_q
            
            # Update remaining columns to compensate
            W[:, col+1:] -= error.unsqueeze(1) @ H_inv[col, col+1:].unsqueeze(0)
            
            # Store quantized value
            W[:, col] = W_col_q
    
    return W
```

### Mixed-Precision Activation Quantization

```python
def quantize_mixed_precision(X, high_fraction=0.125):
    """
    X: activation tensor (batch, seq, hidden_dim)
    high_fraction: fraction of dims in high precision
    """
    hidden_dim = X.shape[-1]
    high_bits_length = int(high_fraction * hidden_dim)
    
    # Split
    X_mid  = X[..., :hidden_dim - high_bits_length]
    X_high = X[..., hidden_dim - high_bits_length:]
    
    # Quantize mid precision (4-bit)
    scale_mid = X_mid.abs().amax(dim=-1, keepdim=True) / 7
    X_mid_q = (X_mid / scale_mid).round().clamp(-8, 7) * scale_mid
    
    # Quantize high precision (8-bit)
    scale_high = X_high.abs().amax(dim=-1, keepdim=True) / 127
    X_high_q = (X_high / scale_high).round().clamp(-128, 127) * scale_high
    
    # Concatenate
    return torch.cat([X_mid_q, X_high_q], dim=-1)
```

## Summary

The ResQ PTQ flow achieves efficient 4-bit quantization through:

1. **PCA-based mixed precision**: Allocate bits optimally based on variance
2. **Rotation-based outlier suppression**: Random rotations spread out outliers
3. **GPTQ weight quantization**: Second-order optimization minimizes error
4. **Hadamard transforms**: Further spread outliers in MLP outputs
5. **KV cache quantization**: Reduce memory for long sequences

Result: **4x memory reduction** with **minimal accuracy loss** and **2.4x speedup**
