# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone R1 Rotation Matrix Extractor for ResQ.

This module extracts the R1 rotation matrix from a Hugging Face model
by running calibration inference to collect activations, performing PCA,
and generating the final rotation matrix.

The R1 rotation is computed as: R1 = matmul(U_attn, block_diag(R1_1, R1_2))
where:
- U_attn: eigenvectors from PCA on covariance matrices of attention and MLP inputs
- R1_1: random orthogonal matrix for main precision channels
- R1_2: random orthogonal matrix for high precision channels

Example usage:
    from transformers import AutoModelForCausalLM
    from r1_rotation_extractor import R1RotationExtractor
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.5B")
    
    extractor = R1RotationExtractor(
        model=model,
        high_fraction=0.125,  # 1/8 of hidden dimension for high precision
        nsamples=128,
        seqlen=2048,
        seed=42
    )
    
    # Extract the rotation matrix
    R1 = extractor.extract()
    
    # Save for later use
    extractor.save("rotation_matrices.pt")
    
    # Load from file
    extractor.load("rotation_matrices.pt")
    R1 = extractor.get_final_r1()
"""

import os
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def random_orthogonal_matrix(size: int, device: str = "cuda") -> torch.Tensor:
    """
    Generate a random orthogonal matrix of the specified size.
    
    Uses QR decomposition to obtain an orthogonal matrix from a random matrix.
    
    Args:
        size: The size of the matrix (size x size).
        device: Device to place the matrix on.
    
    Returns:
        An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def perform_eigen_decomp(cov_matrix: torch.Tensor) -> tuple:
    """
    Perform eigen decomposition on covariance matrix.
    
    Returns sorted eigenvalues and eigenvectors (ascending order).
    
    Args:
        cov_matrix: Covariance matrix to decompose.
        
    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in ascending order.
    """
    device = cov_matrix.device
    H = cov_matrix.to(device)
    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=H.device)
    H[diag, diag] = H[diag, diag] + damp
    X = torch.linalg.eigh(H.to(torch.float64))
    index = torch.argsort(X[0])
    eigenvalues = X[0][index]
    eigenvectors = X[1][:, index]
    return eigenvalues, eigenvectors


@dataclass
class R1RotationConfig:
    """Configuration for R1 rotation extraction."""
    high_fraction: float = 0.125  # Fraction of channels for high precision (default 1/8)
    nsamples: int = 128  # Number of calibration samples
    seqlen: int = 2048  # Sequence length for calibration
    seed: int = 42  # Random seed
    calib_dataset: str = "wikitext"  # Calibration dataset
    device: str = "cuda"  # Device for computation


class R1RotationExtractor:
    """
    Extracts the R1 rotation matrix for ResQ quantization.
    
    This class implements the core functionality of get_basis() from get_basis.py,
    specifically for the full_shared configuration to generate the R1 rotation matrix.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_name: Optional[str] = None,
        high_fraction: float = 0.125,
        nsamples: int = 128,
        seqlen: int = 2048,
        seed: int = 42,
        calib_dataset: str = "wikitext",
        device: str = "cuda",
    ):
        """
        Initialize the R1 rotation extractor.
        
        Args:
            model: Pre-loaded Hugging Face model. If None, model_name must be provided.
            model_name: Model name/path to load from Hugging Face. Ignored if model is provided.
            high_fraction: Fraction of hidden dimension for high precision (default 0.125 = 1/8).
            nsamples: Number of calibration samples.
            seqlen: Sequence length for calibration.
            seed: Random seed for reproducibility.
            calib_dataset: Dataset for calibration ("wikitext", "c4", "ptb", "alpaca").
            device: Device for computation ("cuda" or "cpu").
        """
        self.config = R1RotationConfig(
            high_fraction=high_fraction,
            nsamples=nsamples,
            seqlen=seqlen,
            seed=seed,
            calib_dataset=calib_dataset,
            device=device,
        )
        
        if model is not None:
            self.model = model
            self.model_name = getattr(model.config, '_name_or_path', 'unknown')
        elif model_name is not None:
            self.model_name = model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
        else:
            raise ValueError("Either model or model_name must be provided")
        
        self.model.eval()
        self.model.seqlen = seqlen
        
        # Internal state
        self._U_attn: Optional[torch.Tensor] = None  # Eigenvectors (basis_dict["attn_mlp"])
        self._eigenvalues: Optional[torch.Tensor] = None  # Eigenvalues for analysis
        self._R1_1: Optional[torch.Tensor] = None  # Random orthogonal matrix for main channels
        self._R1_2: Optional[torch.Tensor] = None  # Random orthogonal matrix for high precision
        self._R1: Optional[torch.Tensor] = None  # Final R1 rotation matrix
        
    def _get_calibration_data(self, tokenizer) -> list:
        """Get calibration data from the specified dataset."""
        import random
        import datasets
        
        random.seed(self.config.seed)
        seqlen = self.config.seqlen
        nsamples = self.config.nsamples
        
        if "wikitext" in self.config.calib_dataset:
            traindata = datasets.load_dataset(
                "Salesforce/wikitext", "wikitext-2-raw-v1"
            )["train"]
            trainenc = tokenizer(
                text="\n\n".join(traindata["text"]), return_tensors="pt"
            )
        elif "c4" in self.config.calib_dataset:
            traindata = datasets.load_dataset(
                "allenai/c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train",
            )
            # For c4, we need to handle it differently
            trainloader = []
            for _ in range(nsamples):
                while True:
                    i = random.randint(0, len(traindata) - 1)
                    trainenc = tokenizer(
                        text=traindata[i]["text"], return_tensors="pt"
                    )
                    if trainenc.input_ids.shape[1] > seqlen:
                        break
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))
            return trainloader
        else:
            # Default to wikitext
            traindata = datasets.load_dataset(
                "Salesforce/wikitext", "wikitext-2-raw-v1"
            )["train"]
            trainenc = tokenizer(
                text="\n\n".join(traindata["text"]), return_tensors="pt"
            )
        
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader
    
    def _fuse_layer_norms(self) -> None:
        """Fuse layer norms into adjacent linear layers."""
        # Embedding fusion
        for W in [self.model.model.embed_tokens]:
            W_ = W.weight.data.double()
            W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
        layers = list(self.model.model.layers)
        
        for layer in layers:
            # Fuse post_attention_layernorm into MLP
            self._fuse_ln_linear(
                layer.post_attention_layernorm, 
                [layer.mlp.up_proj, layer.mlp.gate_proj]
            )
            # Fuse input_layernorm into attention
            if hasattr(layer, "self_attn"):
                self._fuse_ln_linear(
                    layer.input_layernorm,
                    [
                        layer.self_attn.q_proj,
                        layer.self_attn.k_proj,
                        layer.self_attn.v_proj,
                    ],
                )
            
            # Set layernorm weights to ones
            W_norm = layer.post_attention_layernorm.weight.data
            layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
            W_norm = layer.input_layernorm.weight.data
            layer.input_layernorm.weight.data = torch.ones_like(W_norm)
        
        # Fuse final norm
        self._fuse_ln_linear(
            self.model.model.norm,
            [self.model.lm_head],
        )
        W_norm = self.model.model.norm.weight.data
        self.model.model.norm.weight.data = torch.ones_like(W_norm)
    
    def _fuse_ln_linear(
        self, 
        layernorm: nn.Module, 
        linear_layers: list
    ) -> None:
        """Fuse layernorm weights into adjacent linear layers."""
        for linear in linear_layers:
            linear_dtype = linear.weight.dtype
            W_ = linear.weight.data.double()
            linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
            
            if hasattr(layernorm, "bias") and layernorm.bias is not None:
                if linear.bias is None:
                    linear.bias = nn.Parameter(
                        torch.zeros(linear.out_features, dtype=linear_dtype)
                    )
                linear.bias.data = linear.bias.data.double() + torch.matmul(
                    W_, layernorm.bias.double()
                )
                linear.bias.data = linear.bias.data.to(linear_dtype)
    
    @torch.no_grad()
    def extract(self) -> torch.Tensor:
        """
        Extract the R1 rotation matrix from the model.
        
        This method:
        1. Fuses layer norms into adjacent linear layers
        2. Runs calibration inference to collect activations
        3. Computes covariance matrices for attention and MLP inputs
        4. Performs PCA to get eigenvectors (U_attn)
        5. Generates random orthogonal matrices R1_1 and R1_2
        6. Computes final R1 = U_attn @ block_diag(R1_1, R1_2)
        
        Returns:
            The final R1 rotation matrix.
        """
        transformers.set_seed(self.config.seed)
        device = self.config.device
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.config.seqlen,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
        )
        
        # Fuse layer norms
        self._fuse_layer_norms()
        
        self.model.config.use_cache = False
        seqlen = self.config.seqlen
        
        # Get calibration data
        train_data = self._get_calibration_data(tokenizer)
        nbatches = len(train_data)
        
        layers = self.model.model.layers
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(device)
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.to(device)
        
        layers[0] = layers[0].to(device)
        
        dtype = next(iter(self.model.parameters())).dtype
        
        # Capture inputs to first layer
        inps = [0] * nbatches
        cache = {"i": 0, "attention_mask": None}
        
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            
            def forward(self, inp, **kwargs):
                inps[cache["i"]] = inp
                cache["i"] += 1
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                if "position_embeddings" in kwargs:
                    cache["position_embeddings"] = kwargs["position_embeddings"]
                else:
                    cache["position_embeddings"] = None
                raise ValueError
        
        layers[0] = Catcher(layers[0])
        for i in range(nbatches):
            batch = train_data[i][0].to(device)
            try:
                self.model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
        
        self.model.model.embed_tokens = self.model.model.embed_tokens.cpu()
        if hasattr(self.model.model, "rotary_emb"):
            self.model.model.rotary_emb = self.model.model.rotary_emb.cpu()
        
        position_ids = cache["position_ids"]
        position_embeddings = cache["position_embeddings"]
        attention_mask = cache["attention_mask"]
        
        torch.cuda.empty_cache()
        outs = [0] * nbatches
        
        hidden_dim = self.model.config.hidden_size
        nlayers = len(layers)
        high_length_hidden = int(self.config.high_fraction * hidden_dim)
        
        # Initialize covariance matrices
        H_attn = torch.zeros((nlayers, hidden_dim, hidden_dim), device=device)
        H_mlp = torch.zeros((nlayers, hidden_dim, hidden_dim), device=device)
        
        # Storage for captured activations (using list to avoid global variables)
        activation_storage = {'input_up_proj': None, 'input_qkv_proj': None}
        
        # Collect covariance matrices
        for i in tqdm(range(nlayers), desc="Collecting covariance matrices"):
            layer = layers[i].to(device)
            
            hooks = []
            
            def hook_fn_upproj(module, input, output, storage=activation_storage):
                storage['input_up_proj'] = input[0]
            
            def hook_fn_qproj(module, input, output, storage=activation_storage):
                storage['input_qkv_proj'] = input[0]
            
            hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn_upproj))
            if hasattr(layer, "self_attn"):
                hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj))
            
            for j in range(nbatches):
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
                
                # Accumulate covariance matrices
                H_mlp[i] += torch.sum(
                    activation_storage['input_up_proj'].double().mT @ activation_storage['input_up_proj'].double(), dim=0
                )
                H_attn[i] += torch.sum(
                    activation_storage['input_qkv_proj'].double().mT @ activation_storage['input_qkv_proj'].double(), dim=0
                )
            
            for hook in hooks:
                hook.remove()
            
            layers[i] = layers[i].cpu()
            torch.cuda.empty_cache()
            inps, outs = outs, inps
        
        # Perform eigen decomposition (full_shared configuration)
        # Divide by 2 because we average over both attention and MLP covariance matrices
        NUM_COV_MATRICES = 2  # H_attn and H_mlp
        cov_matrix = (H_attn.sum(0) + H_mlp.sum(0)) / (NUM_COV_MATRICES * nbatches * nlayers * seqlen)
        self._eigenvalues, self._U_attn = perform_eigen_decomp(cov_matrix)
        
        # Generate random orthogonal matrices
        r1_1_size = hidden_dim - high_length_hidden
        self._R1_1 = random_orthogonal_matrix(r1_1_size, device)
        self._R1_2 = random_orthogonal_matrix(high_length_hidden, device)
        
        # Compute final R1 = U_attn @ block_diag(R1_1, R1_2)
        R1_block = torch.block_diag(self._R1_1, self._R1_2)
        self._R1 = torch.matmul(self._U_attn.to(device), R1_block)
        
        torch.cuda.empty_cache()
        
        return self._R1
    
    def get_final_r1(self) -> torch.Tensor:
        """
        Get the final R1 rotation matrix.
        
        Returns:
            The R1 rotation matrix. Raises ValueError if not yet extracted.
        """
        if self._R1 is None:
            raise ValueError("R1 has not been extracted yet. Call extract() first.")
        return self._R1
    
    def get_components(self) -> Dict[str, torch.Tensor]:
        """
        Get all component matrices.
        
        Returns:
            Dictionary containing:
            - 'U_attn': Eigenvector matrix from PCA
            - 'R1_1': Random orthogonal matrix for main channels  
            - 'R1_2': Random orthogonal matrix for high precision channels
            - 'R1': Final rotation matrix
            - 'eigenvalues': Eigenvalues from PCA (for analysis)
        """
        if self._R1 is None:
            raise ValueError("Components have not been extracted yet. Call extract() first.")
        
        return {
            'U_attn': self._U_attn,
            'R1_1': self._R1_1,
            'R1_2': self._R1_2,
            'R1': self._R1,
            'eigenvalues': self._eigenvalues,
        }
    
    def save(self, path: str) -> None:
        """
        Save the rotation matrices to a file.
        
        Args:
            path: Path to save the rotation matrices.
        """
        if self._R1 is None:
            raise ValueError("Nothing to save. Call extract() first.")
        
        save_dict = {
            'U_attn': self._U_attn.cpu(),
            'R1_1': self._R1_1.cpu(),
            'R1_2': self._R1_2.cpu(),
            'R1': self._R1.cpu(),
            'eigenvalues': self._eigenvalues.cpu(),
            'config': {
                'high_fraction': self.config.high_fraction,
                'nsamples': self.config.nsamples,
                'seqlen': self.config.seqlen,
                'seed': self.config.seed,
                'calib_dataset': self.config.calib_dataset,
                'model_name': self.model_name,
            }
        }
        
        dir_path = os.path.dirname(path) or '.'
        os.makedirs(dir_path, exist_ok=True)
        torch.save(save_dict, path)
        print(f"Rotation matrices saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load rotation matrices from a file.
        
        Args:
            path: Path to load the rotation matrices from.
        """
        loaded = torch.load(path, map_location='cpu')
        
        self._U_attn = loaded['U_attn']
        self._R1_1 = loaded['R1_1']
        self._R1_2 = loaded['R1_2']
        self._R1 = loaded['R1']
        self._eigenvalues = loaded.get('eigenvalues')
        
        # Update config from saved file
        if 'config' in loaded:
            config = loaded['config']
            self.config.high_fraction = config.get('high_fraction', self.config.high_fraction)
            self.config.nsamples = config.get('nsamples', self.config.nsamples)
            self.config.seqlen = config.get('seqlen', self.config.seqlen)
            self.config.seed = config.get('seed', self.config.seed)
            self.config.calib_dataset = config.get('calib_dataset', self.config.calib_dataset)
        
        print(f"Rotation matrices loaded from {path}")
    
    @classmethod
    def from_file(cls, path: str, model: Optional[nn.Module] = None) -> 'R1RotationExtractor':
        """
        Create an extractor instance from a saved file.
        
        Args:
            path: Path to the saved rotation matrices.
            model: Optional model for further operations.
            
        Returns:
            R1RotationExtractor instance with loaded matrices.
        """
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        config = loaded.get('config', {})
        
        # Create a minimal instance with _skip_init flag to avoid validation
        class _LoadedExtractor(cls):
            def __init__(self_inner):
                self_inner.config = R1RotationConfig(
                    high_fraction=config.get('high_fraction', 0.125),
                    nsamples=config.get('nsamples', 128),
                    seqlen=config.get('seqlen', 2048),
                    seed=config.get('seed', 42),
                    calib_dataset=config.get('calib_dataset', 'wikitext'),
                )
                self_inner.model = model
                self_inner.model_name = config.get('model_name', 'unknown')
                self_inner._U_attn = loaded['U_attn']
                self_inner._R1_1 = loaded['R1_1']
                self_inner._R1_2 = loaded['R1_2']
                self_inner._R1 = loaded['R1']
                self_inner._eigenvalues = loaded.get('eigenvalues')
        
        return _LoadedExtractor()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract R1 rotation matrix from a model")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, default="r1_rotation.pt", help="Output path")
    parser.add_argument("--high_fraction", type=float, default=0.125, help="High precision fraction")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--seqlen", type=int, default=2048, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    
    extractor = R1RotationExtractor(
        model=model,
        high_fraction=args.high_fraction,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        device=args.device,
    )
    
    print("Extracting R1 rotation matrix...")
    R1 = extractor.extract()
    
    print(f"R1 shape: {R1.shape}")
    # Verify orthogonality using a small random subset for efficiency
    if R1.shape[0] > 0:
        subset_size = min(64, R1.shape[0])
        R1_subset = R1[:subset_size, :subset_size]
        is_orthogonal = torch.allclose(
            R1_subset @ R1_subset.T, 
            torch.eye(subset_size, device=R1.device, dtype=R1.dtype), 
            atol=1e-6
        )
        print(f"R1 orthogonality check (subset): {is_orthogonal}")
    
    extractor.save(args.output)
