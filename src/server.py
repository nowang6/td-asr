"""FastAPI WebSocket Server for Real-time ASR"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Optional
import click

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

from .asr_engine import ASREngineFactory, ASRResult
from .utils import bytes_to_float32


# Global engine factory
engine_factory: Optional[ASREngineFactory] = None


app = FastAPI(title="TD-ASR WebSocket Server")


@app.get("/")
async def root():
    """Health check endpoint"""
    return JSONResponse({
        "status": "ok",
        "service": "TD-ASR WebSocket Server",
        "version": "0.1.0"
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time ASR
    
    Protocol:
    1. Client connects and sends JSON config message:
       {
           "mode": "2pass",
           "chunk_size": [5, 10, 5],
           "wav_name": "test.wav",
           "wav_format": "pcm",
           "audio_fs": 16000,
           "is_speaking": true
       }
    
    2. Client sends binary audio data (PCM 16-bit)
    
    3. Server sends JSON results:
       {
           "text": "recognized text",
           "is_final": false,
           "timestamp": 0
       }
    
    4. Client sends end message:
       {
           "is_speaking": false
       }
       or
       {
           "is_finished": true
       }
    """
    await websocket.accept()
    
    # Create engine instance for this connection
    if engine_factory is None:
        logger.error("Engine factory not initialized")
        await websocket.close(code=1011, reason="Server not initialized")
        return
    
    engine = engine_factory.create_engine()
    
    # Connection state
    is_started = False
    chunk_size = 800  # Default: 50ms at 16kHz = 800 samples
    audio_buffer = b''
    config_received = False
    
    logger.info(f"WebSocket connection established: {websocket.client}")
    
    try:
        while True:
            # Receive message
            message = await websocket.receive()
            
            if "text" in message:
                # JSON message (config or control)
                try:
                    data = json.loads(message["text"])
                    logger.info(f"Received JSON: {data}")
                    
                    # Handle configuration
                    if "chunk_size" in data and not config_received:
                        # chunk_size is usually [5, 10, 5] from FunASR
                        # This is used for online model initialization
                        # For our implementation, we process in fixed 800 sample chunks
                        config_received = True
                        logger.info(f"Received chunk_size config: {data.get('chunk_size')}")
                    
                    # Handle start
                    if data.get("is_speaking", False) and not is_started:
                        is_started = True
                        config_received = True
                        engine.reset()
                        audio_buffer = b''
                        logger.info("ASR session started")
                    
                    # Handle end
                    if not data.get("is_speaking", True) or data.get("is_finished", False):
                        logger.info("ASR session ending")
                        
                        # Process remaining buffer
                        if len(audio_buffer) > 0:
                            results = engine.process_audio_bytes(audio_buffer, is_finished=True)
                            
                            for result in results:
                                response = {
                                    "text": result.text,
                                    "is_final": result.is_final,
                                    "timestamp": result.timestamp,
                                    "mode": "2pass-offline" if result.is_final else "2pass-online"
                                }
                                await websocket.send_json(response)
                                logger.info(f"Sent result: {response}")
                        else:
                            # Just process end marker
                            results = engine.process_audio_chunk(
                                bytes_to_float32(b''), 
                                is_finished=True
                            )
                            
                            for result in results:
                                response = {
                                    "text": result.text,
                                    "is_final": result.is_final,
                                    "timestamp": result.timestamp,
                                    "mode": "2pass-offline" if result.is_final else "2pass-online"
                                }
                                await websocket.send_json(response)
                        
                        # Reset state
                        is_started = False
                        audio_buffer = b''
                        
                        # Send end message
                        await websocket.send_json({
                            "text": "",
                            "is_final": True,
                            "message": "session ended"
                        })
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    continue
            
            elif "bytes" in message:
                # Binary audio data
                if not is_started or not config_received:
                    logger.warning("Received audio before session started or config received")
                    continue
                
                audio_bytes = message["bytes"]
                audio_buffer += audio_bytes
                
                # Process audio in chunks of 800*2 bytes (800 samples = 50ms @ 16kHz)
                # This matches the C++ implementation (line 534-546 in websocket-server-2pass.cpp)
                chunk_size_bytes = chunk_size * 2  # PCM 16-bit = 2 bytes per sample
                
                if len(audio_buffer) >= chunk_size_bytes:
                    # Calculate how many complete chunks we can process
                    num_chunks = len(audio_buffer) // chunk_size_bytes
                    process_size = num_chunks * chunk_size_bytes
                    
                    # Extract data to process
                    to_process = audio_buffer[:process_size]
                    audio_buffer = audio_buffer[process_size:]
                    
                    # Process chunk
                    results = engine.process_audio_bytes(to_process, is_finished=False)
                    
                    # Send results
                    for result in results:
                        response = {
                            "text": result.text,
                            "is_final": result.is_final,
                            "timestamp": result.timestamp,
                            "mode": "2pass-offline" if result.is_final else "2pass-online"
                        }
                        await websocket.send_json(response)
                        logger.info(f"Sent result: {response}")
            
            else:
                logger.warning(f"Unknown message type: {message}")
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


@click.command()
@click.option('--host', default='0.0.0.0', help='Listen host')
@click.option('--port', default=10095, help='Listen port')
@click.option('--vad-dir', 
              default='models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx',
              help='VAD model directory')
@click.option('--online-model-dir',
              default='models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx',
              help='Online ASR model directory')
@click.option('--offline-model-dir',
              default='models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx',
              help='Offline ASR model directory')
@click.option('--punc-dir',
              default='models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx',
              help='Punctuation model directory')
@click.option('--quantize', default='true', help='Use quantized models')
def main(host: str, port: int, vad_dir: str, online_model_dir: str, 
         offline_model_dir: str, punc_dir: str, quantize: str):
    """Start TD-ASR WebSocket Server"""
    
    global engine_factory
    
    # Setup logging
    logger.info("=" * 60)
    logger.info("TD-ASR WebSocket Server")
    logger.info("=" * 60)
    
    # Resolve paths
    base_dir = Path.cwd()
    vad_path = base_dir / vad_dir
    online_path = base_dir / online_model_dir
    offline_path = base_dir / offline_model_dir
    punc_path = base_dir / punc_dir
    
    # Check model directories
    logger.info(f"VAD model dir: {vad_path}")
    logger.info(f"Online ASR model dir: {online_path}")
    logger.info(f"Offline ASR model dir: {offline_path}")
    logger.info(f"PUNC model dir: {punc_path}")
    
    # Create model config
    model_config = {
        'vad_dir': str(vad_path),
        'online_model_dir': str(online_path),
        'offline_model_dir': str(offline_path),
        'punc_dir': str(punc_path),
        'quantize': quantize,
    }
    
    # Initialize engine factory
    logger.info("Initializing ASR engine factory...")
    try:
        engine_factory = ASREngineFactory(model_config)
        logger.info("✓ ASR engine factory initialized")
    except Exception as e:
        logger.error(f"Failed to initialize engine factory: {e}", exc_info=True)
        return
    
    # Start server
    logger.info(f"Starting server on {host}:{port}")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == '__main__':
    main()

