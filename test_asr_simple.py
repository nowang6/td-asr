#!/usr/bin/env python3
"""Simple test for ASR models"""

import numpy as np
from pathlib import Path
from loguru import logger
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.asr_model import OnlineASRModel, OfflineASRModel


def test_online_asr():
    """Test online ASR model"""
    logger.info("Testing Online ASR Model...")
    
    model_dir = "models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
    
    try:
        model = OnlineASRModel(model_dir, quantize=True)
        logger.info("✓ Model loaded successfully")
        
        # Create dummy audio (1 second, 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        # Extract features
        features = model.extract_features(dummy_audio)
        logger.info(f"✓ Features extracted: shape={features.shape}")
        
        # Run encoder
        encoder_out, cache = model.encode(features)
        logger.info(f"✓ Encoder output: shape={encoder_out.shape}")
        
        # Run decoder
        token_ids = model.decode(encoder_out)
        logger.info(f"✓ Decoder output: {len(token_ids)} tokens")
        logger.info(f"  Token IDs: {token_ids[:20]}...")
        
        # Convert to text
        text = model.tokens_to_text(token_ids)
        logger.info(f"✓ Text: '{text}'")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        return False


def test_offline_asr():
    """Test offline ASR model"""
    logger.info("\nTesting Offline ASR Model...")
    
    model_dir = "models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
    
    try:
        model = OfflineASRModel(model_dir, quantize=True)
        logger.info("✓ Model loaded successfully")
        
        # Create dummy audio (1 second, 16kHz)
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
        
        # Extract features
        features = model.extract_features(dummy_audio)
        logger.info(f"✓ Features extracted: shape={features.shape}")
        
        # Run inference
        text = model.infer(features)
        logger.info(f"✓ Text: '{text}'")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("ASR Models Simple Test")
    logger.info("="*60)
    
    result1 = test_online_asr()
    result2 = test_offline_asr()
    
    logger.info("\n" + "="*60)
    if result1 and result2:
        logger.success("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)

