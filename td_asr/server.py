"""FastAPI WebSocket server for 2-pass ASR"""
import asyncio
import json
from typing import Dict, Set, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
import numpy as np

from .asr_engine import TwoPassASREngine
from .audio import bytes_to_float32
from .config import DEFAULT_HOST, DEFAULT_PORT

app = FastAPI(title="TD-ASR Service")

# Global ASR engine factory (shared models)
_engine_factory: Optional[Dict] = None

# Connection management
active_connections: Set[WebSocket] = set()

def init_engine_factory(
    vad_model_dir=None,
    online_asr_model_dir=None,
    offline_asr_model_dir=None,
    punc_model_dir=None,
    quantize=True,
    thread_num=2,
):
    """Initialize ASR engine factory (stores model paths)"""
    global _engine_factory
    _engine_factory = {
        "vad_model_dir": vad_model_dir,
        "online_asr_model_dir": online_asr_model_dir,
        "offline_asr_model_dir": offline_asr_model_dir,
        "punc_model_dir": punc_model_dir,
        "quantize": quantize,
        "thread_num": thread_num,
    }
    logger.info("ASR engine factory initialized")

def create_engine() -> TwoPassASREngine:
    """Create a new ASR engine instance"""
    global _engine_factory
    if _engine_factory is None:
        from .config import VAD_MODEL_DIR, ONLINE_ASR_MODEL_DIR, OFFLINE_ASR_MODEL_DIR, PUNC_MODEL_DIR
        _engine_factory = {
            "vad_model_dir": VAD_MODEL_DIR,
            "online_asr_model_dir": ONLINE_ASR_MODEL_DIR,
            "offline_asr_model_dir": OFFLINE_ASR_MODEL_DIR,
            "punc_model_dir": PUNC_MODEL_DIR,
            "quantize": True,
            "thread_num": 2,
        }
    
    return TwoPassASREngine(
        vad_model_dir=_engine_factory["vad_model_dir"],
        online_asr_model_dir=_engine_factory["online_asr_model_dir"],
        offline_asr_model_dir=_engine_factory["offline_asr_model_dir"],
        punc_model_dir=_engine_factory["punc_model_dir"],
        quantize=_engine_factory["quantize"],
        thread_num=_engine_factory["thread_num"],
    )

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    logger.info("Server started")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Server shutting down")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for ASR"""
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"Client connected. Total connections: {len(active_connections)}")
    
    # Create per-connection ASR engine
    asr_engine = create_engine()
    
    # Per-connection state
    connection_state = {
        "sample_rate": 16000,
        "wav_format": "pcm",
        "wav_name": "",
        "mode": "2pass",
        "audio_buffer": np.array([], dtype=np.float32),
        "is_speaking": False,
    }
    
    try:
        while True:
            # Receive message
            data = await websocket.receive()
            
            if "bytes" in data:
                # Audio data (PCM 16-bit)
                audio_bytes = data["bytes"]
                audio = bytes_to_float32(audio_bytes)
                
                # Resample if needed
                if connection_state["sample_rate"] != 16000:
                    from .audio import resample_audio
                    audio = resample_audio(
                        audio,
                        connection_state["sample_rate"],
                        16000,
                    )
                
                # Accumulate audio in buffer
                connection_state["audio_buffer"] = np.concatenate([
                    connection_state["audio_buffer"],
                    audio
                ])
                
                # Process audio in chunks (similar to C++ implementation)
                # Process in 800 sample chunks (50ms at 16kHz)
                chunk_size = 800
                while len(connection_state["audio_buffer"]) >= chunk_size:
                    chunk = connection_state["audio_buffer"][:chunk_size]
                    connection_state["audio_buffer"] = connection_state["audio_buffer"][chunk_size:]
                    
                    # Process audio chunk
                    results = asr_engine.process_2pass(chunk, is_final=False)
                    
                    # Send results
                    for result in results:
                        if result.get("text"):  # Only send non-empty results
                            response = {
                                "text": result["text"],
                                "is_final": result.get("is_final", False),
                                "wav_name": connection_state["wav_name"],
                            }
                            await websocket.send_json(response)
            
            elif "text" in data:
                # Text message (JSON configuration or control)
                try:
                    message = json.loads(data["text"])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON message: {e}")
                    continue
                
                # Handle configuration messages (from client initialization)
                if "wav_name" in message:
                    connection_state["wav_name"] = message["wav_name"]
                
                if "mode" in message:
                    connection_state["mode"] = message["mode"]
                
                if "wav_format" in message:
                    connection_state["wav_format"] = message["wav_format"]
                
                if "audio_fs" in message:
                    connection_state["sample_rate"] = message["audio_fs"]
                
                if "is_speaking" in message:
                    connection_state["is_speaking"] = message["is_speaking"]
                
                # Handle final message (is_speaking=false or is_finished=true)
                is_finished = (
                    message.get("is_speaking") == False or
                    message.get("is_finished") == True
                )
                
                if is_finished and connection_state["is_speaking"]:
                    # Process remaining audio buffer
                    if len(connection_state["audio_buffer"]) > 0:
                        final_results = asr_engine.process_2pass(
                            connection_state["audio_buffer"],
                            is_final=True,
                        )
                        connection_state["audio_buffer"] = np.array([], dtype=np.float32)
                        
                        for result in final_results:
                            if result.get("text"):
                                response = {
                                    "text": result["text"],
                                    "is_final": True,
                                    "wav_name": connection_state["wav_name"],
                                }
                                await websocket.send_json(response)
                    
                    # Reset for next session
                    asr_engine.reset()
                    connection_state["is_speaking"] = False
                    connection_state["audio_buffer"] = np.array([], dtype=np.float32)
                
                # Handle legacy control messages
                elif message.get("mode") == "close":
                    # Final processing
                    if len(connection_state["audio_buffer"]) > 0:
                        final_results = asr_engine.process_2pass(
                            connection_state["audio_buffer"],
                            is_final=True,
                        )
                        for result in final_results:
                            response = {
                                "text": result["text"],
                                "is_final": True,
                                "wav_name": connection_state["wav_name"],
                            }
                            await websocket.send_json(response)
                    
                    asr_engine.reset()
                    break
                
                elif message.get("mode") == "reset":
                    # Reset connection
                    asr_engine.reset()
                    connection_state["audio_buffer"] = np.array([], dtype=np.float32)
                    await websocket.send_json({"status": "reset"})
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in websocket: {e}", exc_info=True)
    finally:
        active_connections.discard(websocket)
        asr_engine.reset()
        logger.info(f"Connection closed. Remaining: {len(active_connections)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "connections": len(active_connections),
        "engine_factory_initialized": _engine_factory is not None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)

