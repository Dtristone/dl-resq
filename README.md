
# ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals
This repository contains the code for the submitted paper ResQ



## Abstract
Post-training quantization (PTQ) of large language models (LLMs) holds the promise in reducing the prohibitive computational cost at inference time. Quantization of all weight, activation and key-value (KV) cache tensors to 4-bit without significantly degrading generalizability is challenging, due to the high quantization error caused by extreme outliers in activations. To tackle this problem, we propose **ResQ**, a PTQ method that pushes further the state-of-the-art. By means of principal component analysis (PCA), it identifies a low-rank subspace (in practice 1/8 of the hidden dimension) in which activation variances are highest, and keep the coefficients within this subspace in high precision, e.g.~8-bit, while quantizing the rest to 4-bit. Within each subspace, invariant random rotation is applied to further suppress outliers.  We show that this is a provably optimal mixed precision quantization scheme that minimizes error. With the Llama families of models, we demonstrate that ResQ outperforms recent uniform and mixed precision PTQ methods on a variety of benchmarks, achieving up to 33\% lower perplexity on Wikitext than the next best method SpinQuant, and a 2.4x speedup over 16-bit baseline. 

## Usage
Tested on Python 3.9.19, Torch 2.3.0, CUDA 11.8 on NVIDIA A100 GPU. 
1. Create conda environment
```
conda create -n "resq" python=3.9.19
```
2. Install pytorch locally from [here](https://pytorch.org/get-started/locally/) according to your system specifications. Following should also work
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
3. Install ResQ requirements
```
pip install -r requirements.txt
```
4. Install fast-hadamard-transform library from [here](https://github.com/Dao-AILab/fast-hadamard-transform)
```
git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install .
```
## Running code
Follow these steps to run Llama-3.2-3b with ResQ and evaluate perplexity on wikitext and accuracy on MMLU.
1. Get ResQ projection matrices
```
cd fake_quant
bash 0_get_basis.sh
```
2. Quantize and evaluate the model
```
bash 2_eval_ptq.sh
```

## Documentation

For a detailed understanding of the ResQ PTQ (Post-Training Quantization) flow, please refer to the following documentation files in the `fake_quant/` directory:

- **[PTQ_FLOW_DOCUMENTATION.md](fake_quant/PTQ_FLOW_DOCUMENTATION.md)** - Comprehensive explanation of the entire PTQ pipeline, including:
  - High-level overview of the two-phase process (basis computation and quantization)
  - Detailed breakdown of each step in `ptq.py` and `eval_utils/main.py`
  - Explanation of mixed-precision quantization strategy
  - GPTQ vs RTN weight quantization
  - Activation and KV cache quantization details

- **[PTQ_FLOW_DIAGRAM.md](fake_quant/PTQ_FLOW_DIAGRAM.md)** - Visual flow diagrams showing:
  - Complete pipeline from preprocessing to evaluation
  - Forward pass with and without quantization
  - Mixed-precision dimension partitioning
  - File structure and data flow
  - Key algorithms (GPTQ, mixed-precision quantization)

- **[QUICK_REFERENCE.md](fake_quant/QUICK_REFERENCE.md)** - Quick reference guide with:
  - Quick start commands
  - Important arguments and configuration presets
  - Code entry points and flow
  - Debugging tips and common issues
  - Performance expectations

These documents provide in-depth explanations of how ResQ works under the hood, making it easier to understand, modify, and extend the codebase.

