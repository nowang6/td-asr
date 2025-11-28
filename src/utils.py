"""Utility functions for TD-ASR"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple
import yaml


def bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert PCM 16-bit bytes to float32 array
    
    Args:
        data: PCM 16-bit binary data
        
    Returns:
        Float32 array normalized to [-1, 1]
    """
    # Convert bytes to int16 array
    samples = np.frombuffer(data, dtype=np.int16)
    # Normalize to float32 [-1, 1]
    return samples.astype(np.float32) / 32768.0


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML configuration file
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_cmvn(cmvn_path: Path) -> np.ndarray:
    """Load CMVN statistics from Kaldi Nnet format file
    
    The Kaldi Nnet format contains <AddShift> and <Rescale> sections.
    We extract mean from <AddShift> and std from <Rescale>.
    
    Args:
        cmvn_path: Path to am.mvn file
        
    Returns:
        CMVN array with shape [2, feature_dim], where:
        - cmvn[0] = mean values
        - cmvn[1] = variance values (std^2 or 1/scale^2)
        
    Reference:
        cpp-implement/third_party/onnxruntime/src/fsmn-vad.cpp LoadCmvn()
    """
    with open(cmvn_path, 'r') as f:
        lines = f.readlines()
    
    mean_stats = []
    var_stats = []
    
    current_section = None
    for line in lines:
        line = line.strip()
        
        if '<AddShift>' in line:
            current_section = 'mean'
            continue
        elif '<Rescale>' in line:
            current_section = 'var'
            continue
        elif line.startswith('<LearnRateCoef>'):
            # Extract values from this line
            parts = line.split()
            if current_section == 'mean' and len(parts) > 3:
                # Extract mean values (from index 3 to second-to-last)
                values = [float(x) for x in parts[3:-1] if x.replace('.', '').replace('-', '').replace('e', '').isdigit()]
                mean_stats.extend(values)
            elif current_section == 'var' and len(parts) > 3:
                # Extract variance values (from index 3 to second-to-last)
                values = [float(x) for x in parts[3:-1] if x.replace('.', '').replace('-', '').replace('e', '').isdigit()]
                var_stats.extend(values)
    
    # Return as [mean, var] format
    cmvn = np.array([mean_stats, var_stats], dtype=np.float32)
    return cmvn


def apply_cmvn(features: np.ndarray, cmvn: np.ndarray) -> np.ndarray:
    """Apply CMVN normalization to features
    
    Args:
        features: Feature array with shape [time, feature_dim]
        cmvn: CMVN statistics with shape [2, feature_dim]
        
    Returns:
        Normalized features
    """
    mean = cmvn[0]
    var = cmvn[1]
    
    # Normalize: (x - mean) / sqrt(var)
    # In Kaldi format, var is actually 1/scale, so we multiply
    return (features - mean) * var


def load_tokens(token_path: Path) -> List[str]:
    """Load token list from tokens.txt
    
    Args:
        token_path: Path to tokens.txt
        
    Returns:
        List of tokens
    """
    tokens = []
    with open(token_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 1:
                    tokens.append(parts[0])
    return tokens

