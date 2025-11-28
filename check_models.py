#!/usr/bin/env python3
"""Check if all required models are present"""

from pathlib import Path
from loguru import logger


def check_model(model_dir: Path, model_name: str, required_files: list) -> bool:
    """Check if a model has all required files
    
    Args:
        model_dir: Model directory path
        model_name: Model name for display
        required_files: List of required file names
        
    Returns:
        True if all files exist, False otherwise
    """
    logger.info(f"Checking {model_name}...")
    logger.info(f"  Directory: {model_dir}")
    
    if not model_dir.exists():
        logger.error(f"  ✗ Directory not found")
        return False
    
    all_exist = True
    for file_name in required_files:
        file_path = model_dir / file_name
        if file_path.exists():
            logger.info(f"  ✓ {file_name}")
        else:
            logger.error(f"  ✗ {file_name} NOT FOUND")
            all_exist = False
    
    if all_exist:
        logger.success(f"  {model_name} is ready!")
    else:
        logger.error(f"  {model_name} is incomplete!")
    
    logger.info("")
    return all_exist


def main():
    """Main function to check all models"""
    logger.info("=" * 60)
    logger.info("TD-ASR Model Checker")
    logger.info("=" * 60)
    logger.info("")
    
    base_dir = Path.cwd()
    
    # Define models and their required files
    models = {
        'VAD Model': {
            'dir': base_dir / 'models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx',
            'files': ['model_quant.onnx', 'config.yaml', 'am.mvn']
        },
        'Online ASR Model': {
            'dir': base_dir / 'models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx',
            'files': ['model_quant.onnx', 'decoder_quant.onnx', 'config.yaml', 'am.mvn', 'tokens.json']
        },
        'Offline ASR Model': {
            'dir': base_dir / 'models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx',
            'files': ['model_quant.onnx', 'config.yaml', 'am.mvn', 'tokens.json']
        },
        'Punctuation Model': {
            'dir': base_dir / 'models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx',
            'files': ['model_quant.onnx', 'config.yaml', 'tokens.json']
        },
    }
    
    # Check each model
    results = {}
    for model_name, model_info in models.items():
        results[model_name] = check_model(
            model_info['dir'],
            model_name,
            model_info['files']
        )
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    
    all_ready = True
    for model_name, is_ready in results.items():
        status = "✓ Ready" if is_ready else "✗ Incomplete"
        logger.info(f"{model_name}: {status}")
        if not is_ready:
            all_ready = False
    
    logger.info("")
    
    if all_ready:
        logger.success("All models are ready! You can start the server now.")
        logger.info("Run: python main.py")
        return 0
    else:
        logger.error("Some models are missing. Please download them first.")
        logger.info("Options:")
        logger.info("1. Run: python download_models.py")
        logger.info("2. Or manually download from ModelScope")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

