# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper script to create custom R4 rotation matrices for down_proj.

Usage:
    python create_custom_r4.py --size 14336 --output ./rotation/custom_R4.bin
    python create_custom_r4.py --size 14336 --output ./rotation/custom_R4.bin --seed 42
    python create_custom_r4.py --size 14336 --output ./rotation/custom_R4.bin --type hadamard
"""

import argparse
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hadamard_utils import random_hadamard_matrix


def create_random_orthogonal(size: int, seed: int = None) -> torch.Tensor:
    """Create a random orthogonal matrix using QR decomposition."""
    if seed is not None:
        torch.manual_seed(seed)
    
    random_matrix = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def create_hadamard(size: int) -> torch.Tensor:
    """Create a Hadamard matrix (size must be power of 2)."""
    return random_hadamard_matrix(size, "cuda").to(torch.float64)


def create_identity(size: int) -> torch.Tensor:
    """Create an identity matrix (no rotation)."""
    return torch.eye(size, dtype=torch.float64)


def main():
    parser = argparse.ArgumentParser(description="Create custom R4 rotation matrix for down_proj")
    parser.add_argument("--size", type=int, required=True, 
                        help="Size of the rotation matrix (should match intermediate_size of model)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for the rotation matrix (.bin or .pt)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (only for random type)")
    parser.add_argument("--type", type=str, choices=["random", "hadamard", "identity"], 
                        default="random",
                        help="Type of rotation matrix to create")
    
    args = parser.parse_args()
    
    print(f"Creating {args.type} R4 rotation matrix of size {args.size}x{args.size}")
    
    if args.type == "random":
        R4 = create_random_orthogonal(args.size, args.seed)
        if args.seed:
            print(f"  Using random seed: {args.seed}")
    elif args.type == "hadamard":
        # Check if size is power of 2
        if args.size & (args.size - 1) != 0:
            print(f"Warning: Size {args.size} is not a power of 2, Hadamard may not be exact")
        R4 = create_hadamard(args.size)
    else:  # identity
        R4 = create_identity(args.size)
    
    # Verify orthogonality
    error = torch.norm(R4 @ R4.T - torch.eye(args.size, dtype=torch.float64))
    print(f"  Orthogonality error: {error.item():.2e}")
    
    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save({"R4": R4, "type": args.type, "size": args.size, "seed": args.seed}, args.output)
    print(f"Saved R4 rotation to: {args.output}")
    
    # Print usage hint
    print(f"\nTo use this R4 rotation, add the following flag to your ptq.py command:")
    print(f"  --custom_r4_path {args.output}")


if __name__ == "__main__":
    main()
