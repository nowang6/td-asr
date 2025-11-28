"""Configuration for TD-ASR service"""
from pathlib import Path
from typing import Optional

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"

# Model directories
VAD_MODEL_DIR = MODELS_DIR / "damo" / "speech_fsmn_vad_zh-cn-16k-common-onnx"
ONLINE_ASR_MODEL_DIR = MODELS_DIR / "damo" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx"
OFFLINE_ASR_MODEL_DIR = MODELS_DIR / "damo" / "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
PUNC_MODEL_DIR = MODELS_DIR / "damo" / "punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx"
ITN_MODEL_DIR = MODELS_DIR / "thuduj12" / "fst_itn_zh"

# Audio configuration
SAMPLE_RATE = 16000
FRAME_SHIFT_MS = 10
FRAME_LENGTH_MS = 25
N_MELS = 80

# VAD configuration
VAD_WINDOW_SIZE_MS = 200
VAD_MAX_END_SILENCE_TIME = 800
VAD_MAX_START_SILENCE_TIME = 3000

# Server configuration
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 10095

