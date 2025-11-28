#!/usr/bin/env python3
"""Test script to verify model loading and basic inference"""

import numpy as np
from pathlib import Path
from loguru import logger
import onnxruntime as ort


def test_model_io(model_path: Path, model_name: str):
    """Test a model's input/output"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {model_name}")
    logger.info(f"Path: {model_path}")
    logger.info(f"{'='*60}")
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False
    
    try:
        sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        
        logger.info("\nInputs:")
        for inp in sess.get_inputs():
            logger.info(f"  - {inp.name}: {inp.shape} ({inp.type})")
        
        logger.info("\nOutputs:")
        for out in sess.get_outputs():
            logger.info(f"  - {out.name}: {out.shape} ({out.type})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def main():
    """Test all models"""
    base_dir = Path.cwd()
    
    models = {
        'VAD Model': base_dir / 'models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/model_quant.onnx',
        'Online Encoder': base_dir / 'models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/model_quant.onnx',
        'Online Decoder': base_dir / 'models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/decoder_quant.onnx',
        'Offline ASR': base_dir / 'models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx/model_quant.onnx',
        'Punctuation': base_dir / 'models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx/model_quant.onnx',
    }
    
    results = {}
    for name, path in models.items():
        results[name] = test_model_io(path, name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Summary")
    logger.info(f"{'='*60}")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {name}")
    
    return all(results.values())


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

