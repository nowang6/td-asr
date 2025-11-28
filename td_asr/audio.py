"""Audio preprocessing utilities"""
import numpy as np
from typing import List, Tuple
import struct

def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """Convert PCM bytes to float32 array"""
    # Assume 16-bit PCM
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0

def float32_to_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 array to PCM bytes"""
    # Clip to [-1, 1] and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    samples = (audio * 32768.0).astype(np.int16)
    return samples.tobytes()

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple resampling using linear interpolation"""
    if orig_sr == target_sr:
        return audio
    
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    
    # Linear interpolation
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio)

