#!/usr/bin/env python3
"""Download models from ModelScope for TD-ASR

This script downloads the required ONNX models from ModelScope.
It follows the same approach as the C++ implementation.
"""

import os
import sys
from pathlib import Path
import subprocess
from loguru import logger


# Model configurations
MODELS = {
    'vad': {
        'name': 'damo/speech_fsmn_vad_zh-cn-16k-common-onnx',
        'revision': 'v2.0.4',
    },
    'online_asr': {
        'name': 'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx',
        'revision': 'v2.0.5',
    },
    'offline_asr': {
        'name': 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx',
        'revision': 'v2.0.5',
    },
    'punc': {
        'name': 'damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx',
        'revision': 'v2.0.5',
    },
    'itn': {
        'name': 'thuduj12/fst_itn_zh',
        'revision': 'v1.0.1',
    },
    'lm': {
        'name': 'damo/speech_ngram_lm_zh-cn-ai-wesp-fst',
        'revision': 'v1.0.2',
    },
}


def download_model(model_name: str, model_revision: str, export_dir: Path, export_onnx: bool = True):
    """Download a model from ModelScope
    
    Args:
        model_name: Model name on ModelScope (e.g., 'damo/speech_fsmn_vad_zh-cn-16k-common-onnx')
        model_revision: Model revision (e.g., 'v2.0.4')
        export_dir: Directory to export the model
        export_onnx: Whether to export ONNX model (default: True)
    """
    logger.info(f"Downloading model: {model_name} (revision: {model_revision})")
    
    # Check if model already exists
    model_path = export_dir / model_name
    if model_path.exists():
        logger.info(f"  Model already exists at {model_path}")
        return True
    
    # Construct download command
    cmd = [
        'python', '-m', 'funasr.download.runtime_sdk_download_tool',
        '--type', 'onnx',
        '--quantize', 'True',
        '--model-name', model_name,
        '--export-dir', str(export_dir),
        '--model_revision', model_revision,
    ]
    
    if not export_onnx:
        cmd.extend(['--export', 'False'])
    
    logger.info(f"  Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"  Download command exited with code {result.returncode}")
            logger.warning(f"  stdout: {result.stdout}")
            logger.warning(f"  stderr: {result.stderr}")
            return False
        
        logger.info(f"  ✓ Successfully downloaded {model_name}")
        return True
    
    except Exception as e:
        logger.error(f"  Failed to download {model_name}: {e}")
        return False


def main():
    """Main function to download all models"""
    logger.info("=" * 60)
    logger.info("TD-ASR Model Downloader")
    logger.info("=" * 60)
    
    # Get export directory
    base_dir = Path.cwd()
    export_dir = base_dir / 'models'
    export_dir.mkdir(exist_ok=True)
    
    logger.info(f"Export directory: {export_dir}")
    logger.info("")
    
    # Download models
    success_count = 0
    fail_count = 0
    
    for model_key, model_info in MODELS.items():
        logger.info(f"Downloading {model_key} model...")
        
        export_onnx = model_key not in ['itn', 'lm']  # ITN and LM don't need ONNX export
        
        success = download_model(
            model_info['name'],
            model_info['revision'],
            export_dir,
            export_onnx=export_onnx
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"Download Summary: {success_count} succeeded, {fail_count} failed")
    logger.info("=" * 60)
    
    if fail_count > 0:
        logger.warning("Some models failed to download. You may need to:")
        logger.warning("1. Install FunASR: pip install funasr")
        logger.warning("2. Manually download models from ModelScope")
        logger.warning("3. Place model files in the correct directories")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

