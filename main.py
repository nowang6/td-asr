"""Main entry point for TD-ASR service"""
import click
from pathlib import Path
from loguru import logger
import uvicorn

from td_asr.server import app, init_engine_factory
from td_asr.config import (
    VAD_MODEL_DIR,
    ONLINE_ASR_MODEL_DIR,
    OFFLINE_ASR_MODEL_DIR,
    PUNC_MODEL_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


@click.command()
@click.option(
    "--host",
    default=DEFAULT_HOST,
    help="Host to bind to",
)
@click.option(
    "--port",
    default=DEFAULT_PORT,
    type=int,
    help="Port to bind to",
)
@click.option(
    "--vad-model-dir",
    type=click.Path(exists=True, path_type=Path),
    default=VAD_MODEL_DIR,
    help="VAD model directory",
)
@click.option(
    "--online-asr-model-dir",
    type=click.Path(exists=True, path_type=Path),
    default=ONLINE_ASR_MODEL_DIR,
    help="Online ASR model directory",
)
@click.option(
    "--offline-asr-model-dir",
    type=click.Path(exists=True, path_type=Path),
    default=OFFLINE_ASR_MODEL_DIR,
    help="Offline ASR model directory",
)
@click.option(
    "--punc-model-dir",
    type=click.Path(exists=True, path_type=Path),
    default=PUNC_MODEL_DIR,
    help="Punctuation model directory",
)
@click.option(
    "--quantize/--no-quantize",
    default=True,
    help="Use quantized models",
)
@click.option(
    "--thread-num",
    default=2,
    type=int,
    help="Number of threads for model inference",
)
def main(
    host: str,
    port: int,
    vad_model_dir: Path,
    online_asr_model_dir: Path,
    offline_asr_model_dir: Path,
    punc_model_dir: Path,
    quantize: bool,
    thread_num: int,
):
    """Start TD-ASR WebSocket server"""
    logger.info("Initializing TD-ASR service...")
    logger.info(f"VAD model: {vad_model_dir}")
    logger.info(f"Online ASR model: {online_asr_model_dir}")
    logger.info(f"Offline ASR model: {offline_asr_model_dir}")
    logger.info(f"PUNC model: {punc_model_dir}")
    
    # Initialize ASR engine factory
    init_engine_factory(
        vad_model_dir=vad_model_dir,
        online_asr_model_dir=online_asr_model_dir,
        offline_asr_model_dir=offline_asr_model_dir,
        punc_model_dir=punc_model_dir,
        quantize=quantize,
        thread_num=thread_num,
    )
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
