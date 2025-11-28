#!/bin/bash
# 启动 TD-ASR WebSocket 服务

# 设置默认参数
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-10095}
VAD_DIR=${VAD_DIR:-models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx}
ONLINE_MODEL_DIR=${ONLINE_MODEL_DIR:-models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx}
OFFLINE_MODEL_DIR=${OFFLINE_MODEL_DIR:-models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx}
PUNC_DIR=${PUNC_DIR:-models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx}
QUANTIZE=${QUANTIZE:-true}

echo "========================================="
echo "TD-ASR WebSocket Server"
echo "========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "VAD: $VAD_DIR"
echo "Online ASR: $ONLINE_MODEL_DIR"
echo "Offline ASR: $OFFLINE_MODEL_DIR"
echo "PUNC: $PUNC_DIR"
echo "Quantize: $QUANTIZE"
echo "========================================="

python main.py \
    --host "$HOST" \
    --port "$PORT" \
    --vad-dir "$VAD_DIR" \
    --online-model-dir "$ONLINE_MODEL_DIR" \
    --offline-model-dir "$OFFLINE_MODEL_DIR" \
    --punc-dir "$PUNC_DIR" \
    --quantize "$QUANTIZE"

