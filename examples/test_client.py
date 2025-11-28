"""Simple test client for TD-ASR service"""
import asyncio
import websockets
import json
import sys
from pathlib import Path

# 默认音频文件路径
DEFAULT_AUDIO_FILE = "data/张三丰.wav"

async def test_client(uri: str, audio_file: str = None):
    """Test WebSocket client"""
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send configuration (optional)
            config = {
                "sample_rate": 16000,
                "wav_format": "pcm"
            }
            await websocket.send(json.dumps(config))
            
            # Send audio file
            audio_path = audio_file or DEFAULT_AUDIO_FILE
            try:
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                    print(f"Sending audio data from {audio_path}: {len(audio_data)} bytes")
                    await websocket.send(audio_data)
            except FileNotFoundError:
                print(f"Audio file not found: {audio_path}, sending silence instead")
                # 1 second of silence: 16000 samples * 2 bytes = 32000 bytes
                silence = b'\x00' * 32000
                await websocket.send(silence)
            
            # Wait for results
            print("Waiting for results...")
            try:
                while True:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    result = json.loads(response)
                    print(f"Result: {result['text']}, Final: {result['is_final']}")
                    if result.get('is_final'):
                        break
            except asyncio.TimeoutError:
                print("No response received")
            
            # Send close signal
            await websocket.send(json.dumps({"mode": "close"}))
            print("Sent close signal")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TD-ASR WebSocket client")
    parser.add_argument(
        "--uri",
        default="ws://localhost:10095/ws",
        help="WebSocket server URI"
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=DEFAULT_AUDIO_FILE,
        help="Path to audio file (PCM 16-bit, 16kHz)"
    )
    
    args = parser.parse_args()
    
    asyncio.run(test_client(args.uri, args.audio))

