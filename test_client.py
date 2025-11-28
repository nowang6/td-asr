#!/usr/bin/env python3
"""Simple test client for TD-ASR WebSocket server"""

import asyncio
import json
import wave
from pathlib import Path
import click
import websockets
from loguru import logger


async def send_audio_file(websocket, audio_file: Path, chunk_size: int = 800):
    """Send audio file to WebSocket server
    
    Args:
        websocket: WebSocket connection
        audio_file: Path to WAV file
        chunk_size: Chunk size in samples (default: 800 = 50ms at 16kHz)
    """
    # Open WAV file
    with wave.open(str(audio_file), 'rb') as wf:
        # Check format
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        
        logger.info(f"Audio format: {channels} channels, {sample_width} bytes/sample, {framerate} Hz")
        
        if channels != 1:
            logger.error("Only mono audio is supported")
            return
        
        if sample_width != 2:
            logger.error("Only 16-bit audio is supported")
            return
        
        if framerate != 16000:
            logger.warning(f"Expected 16kHz, got {framerate}Hz")
        
        # Send start message
        start_message = {
            "mode": "2pass",
            "chunk_size": [5, 10, 5],
            "wav_name": audio_file.name,
            "wav_format": "pcm",
            "audio_fs": framerate,
            "is_speaking": True
        }
        await websocket.send(json.dumps(start_message))
        logger.info(f"Sent start message: {start_message}")
        
        # Send audio chunks
        chunk_bytes = chunk_size * 2  # 2 bytes per sample (16-bit)
        total_sent = 0
        
        while True:
            audio_data = wf.readframes(chunk_size)
            if not audio_data:
                break
            
            await websocket.send(audio_data)
            total_sent += len(audio_data)
            
            # Wait a bit to simulate real-time streaming
            await asyncio.sleep(0.05)  # 50ms
        
        logger.info(f"Sent {total_sent} bytes of audio")
        
        # Send end message
        end_message = {
            "is_speaking": False
        }
        await websocket.send(json.dumps(end_message))
        logger.info("Sent end message")


async def receive_results(websocket):
    """Receive and print recognition results
    
    Args:
        websocket: WebSocket connection
    """
    try:
        while True:
            message = await websocket.recv()
            
            # Parse JSON response
            try:
                result = json.loads(message)
                
                mode = result.get('mode', 'unknown')
                text = result.get('text', '')
                is_final = result.get('is_final', False)
                timestamp = result.get('timestamp', 0)
                
                if is_final:
                    logger.success(f"[FINAL] {text}")
                else:
                    logger.info(f"[PARTIAL] {text}")
                
                # Check for end message
                if result.get('message') == 'session ended':
                    logger.info("Session ended")
                    break
                    
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
    
    except websockets.exceptions.ConnectionClosed:
        logger.info("Connection closed")


async def test_websocket(server_url: str, audio_file: Path):
    """Test WebSocket connection
    
    Args:
        server_url: WebSocket server URL
        audio_file: Path to audio file
    """
    logger.info(f"Connecting to {server_url}")
    
    try:
        async with websockets.connect(server_url) as websocket:
            logger.info("Connected successfully")
            
            # Create tasks for sending and receiving
            send_task = asyncio.create_task(send_audio_file(websocket, audio_file))
            receive_task = asyncio.create_task(receive_results(websocket))
            
            # Wait for both tasks to complete
            await asyncio.gather(send_task, receive_task)
            
            logger.info("Test completed")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


@click.command()
@click.option('--server', default='ws://localhost:10095/ws', help='WebSocket server URL')
@click.option('--audio', required=True, type=click.Path(exists=True), help='Path to audio file (WAV)')
def main(server: str, audio: str):
    """Test client for TD-ASR WebSocket server"""
    audio_file = Path(audio)
    
    if not audio_file.exists():
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    if audio_file.suffix.lower() != '.wav':
        logger.error("Only WAV files are supported")
        return
    
    # Run test
    asyncio.run(test_websocket(server, audio_file))


if __name__ == '__main__':
    main()

