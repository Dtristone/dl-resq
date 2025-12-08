# ResQ PTQ Quick Reference Guide

## Quick Start

### 1. Compute Basis Matrices (One-time per model)
```bash
cd fake_quant
bash 0_get_basis.sh
```
**Output**: 
- `rotation/U-wikitext-512-{model}.bin` (PCA basis)
- `rotation/R-high-{h}-low-{l}-sparse-{s}-{model}.bin` (rotation matrices)

### 2. Run PTQ and Evaluate
```bash
bash 2_eval_ptq.sh
```
**Output**: Model perplexity and task accuracy

## Key Files Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `ptq.py` | Main entry point | `train()`, `evaluate()` |
| `eval_utils/main.py` | Core PTQ logic | `ptq_model()` |
| `get_basis.py` | Compute U and R | `get_outlier_rotations()` |
| `utils/quant_utils.py` | Quantizers | `ActQuantizer`, `WeightQuantizer` |
| `eval_utils/rotation_utils.py` | Rotation ops | `fuse_basis_to_model()`, `rearrange_columns()` |
| `eval_utils/gptq_utils.py` | Weight quantization | `gptq_fwrd()`, `rtn_fwrd()` |

## Important Arguments

### Model Selection
```bash
--input_model meta-llama/Llama-3.2-1B    # Model to quantize
```

### Quantization Bits
```bash
--w_bits 4       # Weight bits (or 16 for no weight quantization)
--a_bits 4       # Activation bits
--k_bits 4       # Key cache bits
--v_bits 4       # Value cache bits
```

### Mixed Precision
```bash
--high_bits 8          # High precision bits (for outliers)
--low_bits 2           # Low precision bits (optional)
--high_fraction 0.125  # Fraction in high precision (12.5%)
--low_fraction 0.0     # Fraction in low precision (0%)
```

### Rotation Mode
```bash
--rotate_mode "resq"   # Use ResQ (recommended)
                       # Options: "resq", "quik", "hadamard", "random", "none"

--optimized_rotation_path ./rotation/R-*.bin   # Path to R matrices
--optimized_basis_path ./rotation/U-*.bin      # Path to U matrices
```

### Weight Quantization Method
```bash
--w_rtn False    # Use GPTQ (False) or RTN (True)
--nsamples 128   # Number of calibration samples for GPTQ
```

### Evaluation
```bash
--tasks "mmlu,boolq,piqa,hellaswag"    # Tasks to evaluate
--num_fewshot 0                        # Number of few-shot examples
--bsz 1                                # Batch size
```

## Configuration Presets

### Full ResQ W16A4K4V4 (Recommended)
```bash
--w_bits 16 --a_bits 4 --k_bits 4 --v_bits 4 \
--high_bits 8 --high_fraction 0.125 \
--rotate_mode "resq" \
--optimized_rotation_path ./rotation/R-*.bin \
--optimized_basis_path ./rotation/U-*.bin
```

### ResQ W4A4K4V4 (Maximum Compression)
```bash
--w_bits 4 --a_bits 4 --k_bits 4 --v_bits 4 \
--high_bits 8 --high_fraction 0.125 \
--rotate_mode "resq" \
--w_rtn False \
--optimized_rotation_path ./rotation/R-*.bin \
--optimized_basis_path ./rotation/U-*.bin
```

### Baseline (No Quantization)
```bash
--w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 \
--rotate_mode "none"
```

## Code Entry Points

### Entry Point 1: `ptq.py::train()`
```python
# Main workflow:
train()
├─ Load model configuration
├─ Load model (LlamaForCausalLM / Qwen2ForCausalLM)
├─ Call ptq_model(args, model, model_args)  # Apply quantization
├─ Load tokenizer
└─ evaluate(model, tokenizer, args)         # Test the model
```

### Entry Point 2: `eval_utils/main.py::ptq_model()`
```python
# Quantization steps:
ptq_model(args, model, model_args)
├─ Step 1: Rotation & Basis Transformation
│  ├─ fuse_layer_norms(model)
│  ├─ fuse_basis_to_model(model, args)      # Apply U
│  ├─ rearrange_columns(model, args)        # Sort by variance
│  └─ Configure Hadamard for down_proj
├─ Step 2: Weight Quantization
│  ├─ gptq_fwrd(model, trainloader, args)   # GPTQ
│  └─ rtn_fwrd(model, args)                 # RTN
├─ Step 3: Activation Quantization Setup
│  └─ Configure quantizers for each layer
└─ Step 4: KV Cache Quantization
   └─ Add rotation wrappers after RoPE
```

## Understanding the Code Flow

### 1. Where is the quantization applied?

**Activation Quantization**: Applied during forward pass
- Location: `utils/quant_utils.py::ActQuantizer.forward()`
- Called by: `ActQuantWrapper` wrapping each linear layer
- When: Every time activations pass through a layer

**Weight Quantization**: Applied once during `ptq_model()`
- GPTQ: `eval_utils/gptq_utils.py::gptq_fwrd()`
- RTN: `eval_utils/gptq_utils.py::rtn_fwrd()`
- When: During model preparation, before evaluation

**KV Cache Quantization**: Applied during forward pass
- Location: `eval_utils/rotation_utils.py::QKRotationWrapper`
- When: After RoPE, before storing in cache

### 2. Where are the rotation matrices used?

**Basis U**: 
- Fused into weights: `eval_utils/rotation_utils.py::fuse_basis_to_model()`
- Transforms: `W' = W @ U`

**Rotation R**:
- For KV cache: Combined with U as `k_rotation = U @ R`
- Applied in: `QKRotationWrapper` after RoPE

### 3. Where is mixed precision configured?

**Configuration**: `eval_utils/main.py::ptq_model()` lines 80-166
- Calculates `high_bits_length` and `low_bits_length`
- Sets different precision for different layers

**Applied**: `utils/quant_utils.py::ActQuantizer.forward()`
- Splits activations into low/mid/high precision parts
- Quantizes each part with different bit-widths

## Debugging Tips

### 1. Check if rotation is applied
```python
# In ptq.py after ptq_model():
for name, param in model.named_parameters():
    if 'basis_change' in name:
        print(f"{name}: {param.shape}")
```

### 2. Check quantizer configuration
```python
# After ptq_model():
from utils import quant_utils
qlayers = quant_utils.find_qlayers(model)
for name, layer in qlayers.items():
    if hasattr(layer, 'quantizer'):
        print(f"{name}: {layer.quantizer.bits} bits, "
              f"high: {layer.quantizer.high_bits_length}, "
              f"low: {layer.quantizer.low_bits_length}")
```

### 3. Verify rotation matrices exist
```bash
ls -lh rotation/
# Should see:
# U-wikitext-512-{model}.bin
# R-high-0.125-low-0.0-sparse-0.0-{model}.bin
```

### 4. Check model size
```python
# Before and after quantization:
def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2  # MB

print(f"Model size: {get_model_size(model):.2f} MB")
```

## Common Issues

### Issue 1: "rotation matrices not found"
**Solution**: Run `bash 0_get_basis.sh` first to generate U and R matrices

### Issue 2: "Out of memory during GPTQ"
**Solution**: Reduce `--nsamples` (e.g., from 512 to 128)

### Issue 3: "High perplexity after quantization"
**Solution**: 
- Check `--high_fraction` (increase to 0.25 for more high-precision dims)
- Verify rotation matrices are loaded correctly
- Try GPTQ instead of RTN for weights

### Issue 4: "Slow evaluation"
**Solution**: 
- Use `--multigpu` for multi-GPU inference
- Reduce `--limit` to test on fewer samples
- Use `--flash_attn` for faster attention

## Performance Expectations

### WikiText-2 Perplexity (Llama-3.2-1B)
| Configuration | Perplexity | Speedup |
|---------------|-----------|---------|
| FP16 Baseline | ~15.5 | 1.0x |
| W16A4K4V4 (ResQ) | ~16.2 | 2.4x |
| W4A4K4V4 (ResQ) | ~17.5 | 3.0x |

### MMLU Accuracy (Llama-3.2-1B)
| Configuration | Accuracy |
|---------------|----------|
| FP16 Baseline | ~42.8% |
| W16A4K4V4 (ResQ) | ~42.1% |
| W4A4K4V4 (ResQ) | ~40.5% |

*Note: Actual numbers may vary based on calibration data and hyperparameters*

## Further Reading

- **Full Documentation**: `PTQ_FLOW_DOCUMENTATION.md` - Detailed explanation of the PTQ process
- **Flow Diagram**: `PTQ_FLOW_DIAGRAM.md` - Visual diagrams of the PTQ pipeline
- **Paper**: https://arxiv.org/abs/2412.14363 - ResQ paper with theoretical background
- **Code Comments**: Check inline comments in `ptq.py` and `eval_utils/main.py`

## Contact & Citation

If you use ResQ in your research, please cite:
```bibtex
@article{resq2024,
  title={ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals},
  author={[Authors]},
  journal={arXiv preprint arXiv:2412.14363},
  year={2024}
}
```
