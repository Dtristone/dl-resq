# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantization Configuration Utilities for RESQ PTQ model.

This module provides utilities for loading and parsing quantization configurations,
enabling selective rotation and quantization of different modules.

Rotation Matrix Overview:
- R1: Applied to hidden states (embeddings, q/k/v_proj inputs, o_proj output, up/gate_proj inputs, down_proj output)
- R2: Applied to V attention output and O_proj input (per-head rotation in attention computation)
- R3: Applied to K cache (key rotation after RoPE)
- R4: Applied to down_proj input (left side, online Hadamard transform)
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModuleQuantConfig:
    """Configuration for a single module's quantization."""
    enabled: bool = True
    bits: int = 4
    groupsize: int = -1
    asym: bool = True
    clip_ratio: float = 1.0
    # For output quantization (e.g., v_proj output)
    output_quant_enabled: bool = False
    output_bits: int = 16
    output_groupsize: int = -1
    output_asym: bool = True
    # For down_proj special handling
    left_side_high_bits: bool = False
    left_side_bits: int = 16
    right_side_enabled: bool = True
    right_side_bits: int = 4


@dataclass
class RotationConfig:
    """Configuration for rotation matrices."""
    enable_R1: bool = True  # Hidden state rotation (main rotation)
    enable_R2: bool = True  # V/O attention computation rotation (per-head)
    enable_R3: bool = True  # K cache rotation
    enable_R4: bool = True  # Down proj left side (Hadamard)
    custom_R4_path: Optional[str] = None  # Path to custom R4 rotation matrix


@dataclass 
class MixedPrecisionConfig:
    """Configuration for mixed precision quantization."""
    high_bits: int = 8
    low_bits: int = 2
    high_fraction: float = 0.125
    low_fraction: float = 0.0


@dataclass
class OnlineHadamardConfig:
    """Configuration for online Hadamard transform."""
    down_proj: bool = True
    o_proj: bool = False


@dataclass
class QuantizationConfig:
    """Main configuration class for quantization and rotation control."""
    rotation: RotationConfig = field(default_factory=RotationConfig)
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    online_hadamard: OnlineHadamardConfig = field(default_factory=OnlineHadamardConfig)
    
    # Per-module configurations
    q_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    k_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    v_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    o_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    up_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    gate_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    down_proj: ModuleQuantConfig = field(default_factory=ModuleQuantConfig)
    k_cache: ModuleQuantConfig = field(default_factory=lambda: ModuleQuantConfig(enabled=True, bits=4))
    v_cache: ModuleQuantConfig = field(default_factory=lambda: ModuleQuantConfig(enabled=True, bits=4))
    lm_head: ModuleQuantConfig = field(default_factory=lambda: ModuleQuantConfig(enabled=False, bits=16))
    embed_tokens: ModuleQuantConfig = field(default_factory=lambda: ModuleQuantConfig(enabled=False, bits=16))
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizationConfig':
        """Create a QuantizationConfig from a dictionary."""
        config = cls()
        
        # Parse rotation config
        if 'rotation' in config_dict:
            rot_dict = config_dict['rotation']
            config.rotation = RotationConfig(
                enable_R1=rot_dict.get('enable_R1', True),
                enable_R2=rot_dict.get('enable_R2', True),
                enable_R3=rot_dict.get('enable_R3', True),
                enable_R4=rot_dict.get('enable_R4', True),
                custom_R4_path=rot_dict.get('custom_R4_path', None)
            )
        
        # Parse mixed precision config
        if 'mixed_precision' in config_dict:
            mp_dict = config_dict['mixed_precision']
            config.mixed_precision = MixedPrecisionConfig(
                high_bits=mp_dict.get('high_bits', 8),
                low_bits=mp_dict.get('low_bits', 2),
                high_fraction=mp_dict.get('high_fraction', 0.125),
                low_fraction=mp_dict.get('low_fraction', 0.0)
            )
        
        # Parse online Hadamard config
        if 'online_hadamard' in config_dict:
            oh_dict = config_dict['online_hadamard']
            config.online_hadamard = OnlineHadamardConfig(
                down_proj=oh_dict.get('down_proj', True),
                o_proj=oh_dict.get('o_proj', False)
            )
        
        # Parse per-module quantization configs
        if 'quantization' in config_dict:
            quant_dict = config_dict['quantization']
            module_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 
                          'gate_proj', 'down_proj', 'k_cache', 'v_cache', 
                          'lm_head', 'embed_tokens']
            
            for module_name in module_names:
                if module_name in quant_dict:
                    mod_dict = quant_dict[module_name]
                    mod_config = ModuleQuantConfig(
                        enabled=mod_dict.get('enabled', True),
                        bits=mod_dict.get('bits', 4),
                        groupsize=mod_dict.get('groupsize', -1),
                        asym=mod_dict.get('asym', True),
                        clip_ratio=mod_dict.get('clip_ratio', 1.0),
                        output_quant_enabled=mod_dict.get('output_quant_enabled', False),
                        output_bits=mod_dict.get('output_bits', 16),
                        output_groupsize=mod_dict.get('output_groupsize', -1),
                        output_asym=mod_dict.get('output_asym', True),
                        left_side_high_bits=mod_dict.get('left_side_high_bits', False),
                        left_side_bits=mod_dict.get('left_side_bits', 16),
                        right_side_enabled=mod_dict.get('right_side_enabled', True),
                        right_side_bits=mod_dict.get('right_side_bits', 4)
                    )
                    setattr(config, module_name, mod_config)
        
        return config
    
    @classmethod
    def from_json(cls, json_path: str) -> 'QuantizationConfig':
        """Load a QuantizationConfig from a JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'rotation': {
                'enable_R1': self.rotation.enable_R1,
                'enable_R2': self.rotation.enable_R2,
                'enable_R3': self.rotation.enable_R3,
                'enable_R4': self.rotation.enable_R4,
                'custom_R4_path': self.rotation.custom_R4_path
            },
            'mixed_precision': {
                'high_bits': self.mixed_precision.high_bits,
                'low_bits': self.mixed_precision.low_bits,
                'high_fraction': self.mixed_precision.high_fraction,
                'low_fraction': self.mixed_precision.low_fraction
            },
            'online_hadamard': {
                'down_proj': self.online_hadamard.down_proj,
                'o_proj': self.online_hadamard.o_proj
            },
            'quantization': {
                'q_proj': self._module_to_dict(self.q_proj),
                'k_proj': self._module_to_dict(self.k_proj),
                'v_proj': self._module_to_dict(self.v_proj),
                'o_proj': self._module_to_dict(self.o_proj),
                'up_proj': self._module_to_dict(self.up_proj),
                'gate_proj': self._module_to_dict(self.gate_proj),
                'down_proj': self._module_to_dict(self.down_proj),
                'k_cache': self._module_to_dict(self.k_cache),
                'v_cache': self._module_to_dict(self.v_cache),
                'lm_head': self._module_to_dict(self.lm_head),
                'embed_tokens': self._module_to_dict(self.embed_tokens)
            }
        }
    
    def _module_to_dict(self, mod_config: ModuleQuantConfig) -> Dict[str, Any]:
        """Convert a ModuleQuantConfig to a dictionary."""
        return {
            'enabled': mod_config.enabled,
            'bits': mod_config.bits,
            'groupsize': mod_config.groupsize,
            'asym': mod_config.asym,
            'clip_ratio': mod_config.clip_ratio,
            'output_quant_enabled': mod_config.output_quant_enabled,
            'output_bits': mod_config.output_bits,
            'output_groupsize': mod_config.output_groupsize,
            'output_asym': mod_config.output_asym,
            'left_side_high_bits': mod_config.left_side_high_bits,
            'left_side_bits': mod_config.left_side_bits,
            'right_side_enabled': mod_config.right_side_enabled,
            'right_side_bits': mod_config.right_side_bits
        }
    
    def save_to_json(self, json_path: str) -> None:
        """Save the config to a JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    def get_module_config(self, module_name: str) -> Optional[ModuleQuantConfig]:
        """Get the quantization config for a specific module by name."""
        # Extract the base module name from full path (e.g., "model.layers.0.self_attn.q_proj" -> "q_proj")
        for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 
                    'gate_proj', 'down_proj', 'lm_head', 'embed_tokens']:
            if key in module_name:
                return getattr(self, key)
        return None
    
    def should_quantize_input(self, module_name: str) -> bool:
        """Check if the input to a module should be quantized."""
        config = self.get_module_config(module_name)
        if config is None:
            return False
        return config.enabled and config.bits < 16
    
    def get_input_bits(self, module_name: str) -> int:
        """Get the quantization bits for a module's input."""
        config = self.get_module_config(module_name)
        if config is None:
            return 16
        return config.bits if config.enabled else 16
    
    def should_quantize_output(self, module_name: str) -> bool:
        """Check if the output of a module should be quantized (e.g., v_proj output)."""
        config = self.get_module_config(module_name)
        if config is None:
            return False
        return config.output_quant_enabled and config.output_bits < 16


def load_quant_config(config_path: Optional[str] = None) -> QuantizationConfig:
    """
    Load a quantization config from a file, or return default config if no path provided.
    
    Args:
        config_path: Path to the JSON config file. If None, returns default config.
    
    Returns:
        QuantizationConfig instance.
    """
    if config_path is None:
        logger.info("No quant config path provided, using default configuration.")
        return QuantizationConfig()
    
    logger.info(f"Loading quant config from: {config_path}")
    return QuantizationConfig.from_json(config_path)


def create_r1_only_config() -> QuantizationConfig:
    """
    Create a config that only rotates and quantizes R1-related modules.
    
    R1-related modules (quantized):
    - q_proj, k_proj, v_proj (input only), up_proj, gate_proj
    - down_proj output (right side)
    
    Non-R1 modules (kept in high precision):
    - v_proj output (R2-related)
    - o_proj input (R2-related)
    - k_cache (R3-related)
    - v_cache (R2-related)  
    - down_proj input/left side (R4-related)
    """
    config = QuantizationConfig()
    
    # Rotation: only R1
    config.rotation = RotationConfig(
        enable_R1=True,
        enable_R2=False,
        enable_R3=False,
        enable_R4=False
    )
    
    # R1-related modules - enable quantization
    config.q_proj = ModuleQuantConfig(enabled=True, bits=4, asym=True)
    config.k_proj = ModuleQuantConfig(enabled=True, bits=4, asym=True)
    config.v_proj = ModuleQuantConfig(
        enabled=True, bits=4, asym=True,
        output_quant_enabled=False, output_bits=16  # V output is R2-related
    )
    config.up_proj = ModuleQuantConfig(enabled=True, bits=4, asym=True)
    config.gate_proj = ModuleQuantConfig(enabled=True, bits=4, asym=True)
    
    # R2-related - keep in high precision
    config.o_proj = ModuleQuantConfig(enabled=False, bits=16)
    
    # R4-related left side high bits, R1-related right side can be quantized
    config.down_proj = ModuleQuantConfig(
        enabled=False, bits=16,  # Input (left side) not quantized
        left_side_high_bits=True, left_side_bits=16,
        right_side_enabled=True, right_side_bits=4
    )
    
    # KV cache - keep in high precision (R2/R3-related)
    config.k_cache = ModuleQuantConfig(enabled=False, bits=16)
    config.v_cache = ModuleQuantConfig(enabled=False, bits=16)
    
    # Other modules
    config.lm_head = ModuleQuantConfig(enabled=False, bits=16)
    config.embed_tokens = ModuleQuantConfig(enabled=False, bits=16)
    
    # Disable online Hadamard for R4
    config.online_hadamard = OnlineHadamardConfig(down_proj=False, o_proj=False)
    
    return config
