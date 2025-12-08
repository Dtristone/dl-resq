# ResQ PTQ Implementation - Documentation Index

This directory contains the implementation of ResQ's Post-Training Quantization (PTQ) algorithm as described in the paper ["ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals"](https://arxiv.org/abs/2412.14363).

## 📚 Documentation Files

We provide comprehensive documentation to help you understand the PTQ flow in this codebase:

### 1. [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) 📖
**Best for**: Understanding the complete PTQ algorithm in detail

This comprehensive guide covers:
- High-level overview of the two-phase PTQ pipeline
- Detailed step-by-step breakdown of `ptq.py` and `eval_utils/main.py`
- Explanation of rotation and basis transformation
- Weight quantization (GPTQ vs RTN)
- Mixed-precision activation quantization strategy
- KV cache quantization
- Quantization details and implementation specifics

**Recommended if**: You want to deeply understand how ResQ works, modify the algorithm, or implement similar techniques.

### 2. [PTQ_FLOW_DIAGRAM.md](PTQ_FLOW_DIAGRAM.md) 🎨
**Best for**: Visual learners who prefer diagrams

This document provides:
- ASCII-art flow diagrams of the complete pipeline
- Visual comparison of standard vs quantized transformer layers
- Mixed-precision dimension partitioning diagrams
- File structure and data flow visualization
- Simplified algorithm pseudocode

**Recommended if**: You prefer visual representations, want to quickly grasp the overall flow, or need to present the architecture to others.

### 3. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) ⚡
**Best for**: Quick lookup and practical usage

This guide includes:
- Quick start commands
- Important command-line arguments
- Configuration presets for different use cases
- Code entry points and execution flow
- Debugging tips and common issues
- Performance expectations

**Recommended if**: You want to quickly run experiments, troubleshoot issues, or find specific configuration options.

## 🚀 Quick Start

1. **First time setup** - Compute basis matrices:
```bash
bash 0_get_basis.sh
```

2. **Run PTQ and evaluate**:
```bash
bash 2_eval_ptq.sh
```

3. **Want to understand what's happening?** 
   - Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for a quick overview
   - Then dive into [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) for details

## 📂 Main Code Files

| File | Purpose | Documentation Reference |
|------|---------|------------------------|
| `ptq.py` | Main entry point for PTQ | See "Entry Point: train()" in [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) |
| `get_basis.py` | Compute PCA basis and rotation matrices | See "Phase 1: Preprocessing" in [PTQ_FLOW_DIAGRAM.md](PTQ_FLOW_DIAGRAM.md) |
| `eval_utils/main.py` | Core PTQ logic (`ptq_model()`) | See "Core PTQ Function" in [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) |
| `utils/quant_utils.py` | Quantizer implementations | See "Quantizer Implementation" in [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) |
| `eval_utils/rotation_utils.py` | Rotation transformations | See "Step 1: Rotation" in both docs |
| `eval_utils/gptq_utils.py` | GPTQ weight quantization | See "GPTQ Algorithm" in [PTQ_FLOW_DIAGRAM.md](PTQ_FLOW_DIAGRAM.md) |

## 💡 Where to Start?

**If you're new to the codebase:**
1. Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - "Code Entry Points" section
2. Look at the diagrams in [PTQ_FLOW_DIAGRAM.md](PTQ_FLOW_DIAGRAM.md) - "Overall Pipeline"
3. Read the detailed explanation in [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md)

**If you want to modify the code:**
1. Read [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) - understand each step
2. Check inline comments in `ptq.py` and `eval_utils/main.py`
3. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for debugging tips

**If you're debugging an issue:**
1. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - "Common Issues" section
2. Use debugging code snippets from the same guide
3. Refer to [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md) for implementation details

## 🔑 Key Concepts

### Mixed-Precision Quantization
ResQ uses different bit-widths for different parts of activations:
- **High variance dimensions** (12.5%): 8-bit
- **Medium variance dimensions** (87.5%): 4-bit
- **Low variance dimensions** (optional): 2-bit

See the "Mixed-Precision Quantization Strategy" section in [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md).

### Rotation-based Outlier Suppression
ResQ applies PCA basis transformation + random rotation to spread outliers:
- **U matrices**: PCA basis (captures high-variance subspace)
- **R matrices**: Random rotation (further suppresses outliers)

See "Step 1: Rotation & Basis Transformation" in [PTQ_FLOW_DIAGRAM.md](PTQ_FLOW_DIAGRAM.md).

### GPTQ Weight Quantization
Second-order optimization for weight quantization using Hessian information.

See "GPTQ Weight Quantization" in [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md).

## 📊 Expected Results

With Llama-3.2-1B on WikiText-2:
- **FP16 Baseline**: ~15.5 perplexity
- **W16A4K4V4 (ResQ)**: ~16.2 perplexity, 2.4x speedup
- **W4A4K4V4 (ResQ)**: ~17.5 perplexity, 3.0x speedup

See "Performance Expectations" in [QUICK_REFERENCE.md](QUICK_REFERENCE.md).

## 🤝 Contributing

When modifying the code:
1. Keep documentation in sync with code changes
2. Add inline comments for complex logic
3. Update the relevant documentation files
4. Test with the provided scripts

## 📖 Citation

If you use this code, please cite:
```bibtex
@article{resq2024,
  title={ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals},
  author={[Authors]},
  journal={arXiv preprint arXiv:2412.14363},
  year={2024}
}
```

## 📞 Questions?

- **How does PTQ work?** → Read [PTQ_FLOW_DOCUMENTATION.md](PTQ_FLOW_DOCUMENTATION.md)
- **What's the pipeline?** → See diagrams in [PTQ_FLOW_DIAGRAM.md](PTQ_FLOW_DIAGRAM.md)
- **How do I run it?** → Follow [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Where's the code?** → Check inline comments in `ptq.py` and `eval_utils/main.py`

---

**Documentation created**: December 2024  
**Last updated**: December 2024  
**Related paper**: https://arxiv.org/abs/2412.14363
