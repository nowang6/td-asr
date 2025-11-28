# TD-ASR: Two-Pass ASR Service

基于 FastAPI 和 ONNX Runtime 的实时语音听写服务，实现 2-pass（两阶段）语音识别。

## 架构

根据 `design.md`，系统采用两阶段识别：

1. **Online（实时）阶段**：
   - 音频输入 → FSMN-VAD（端点检测） → Paraformer-online（实时识别） → 实时输出（~600ms延迟）

2. **Offline（非实时）阶段**：
   - VAD 尾点触发 → Paraformer-offline（离线识别） → CT-Transformer（标点预测） → ITN（逆文本正规化） → 最终输出

## 安装

```bash
# 安装依赖
uv sync
```

## 使用方法

### 启动服务

```bash
python main.py
```

或者使用自定义参数：

```bash
python main.py \
    --host 0.0.0.0 \
    --port 10095 \
    --thread-num 2
```

### WebSocket 客户端示例

```python
import asyncio
import websockets
import json

async def test_client():
    uri = "ws://localhost:10095/ws"
    async with websockets.connect(uri) as websocket:
        # 发送音频数据（PCM 16-bit, 16kHz）
        # 这里需要替换为实际的音频数据
        audio_data = b''  # 你的音频字节数据
        
        # 发送音频
        await websocket.send(audio_data)
        
        # 接收识别结果
        response = await websocket.recv()
        result = json.loads(response)
        print(f"识别结果: {result['text']}, 是否最终: {result['is_final']}")
        
        # 发送结束信号
        await websocket.send(json.dumps({"mode": "close"}))

asyncio.run(test_client())
```

## 模型结构

服务使用以下模型（位于 `models/` 目录）：

- **VAD**: `damo/speech_fsmn_vad_zh-cn-16k-common-onnx`
- **Online ASR**: `damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx`
- **Offline ASR**: `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx`
- **PUNC**: `damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx`
- **ITN**: `thuduj12/fst_itn_zh`（暂未实现）

## API

### WebSocket 端点

- **URL**: `ws://host:port/ws`
- **协议**: WebSocket

#### 消息格式

**发送音频数据**：
- 类型：二进制（bytes）
- 格式：PCM 16-bit，16kHz，单声道
- 每次发送一个音频块

**发送控制消息**（JSON）：
```json
{
    "mode": "close",  // 结束识别
    "mode": "reset",  // 重置连接
    "sample_rate": 16000,  // 设置采样率
    "wav_format": "pcm"    // 设置音频格式
}
```

**接收识别结果**（JSON）：
```json
{
    "text": "识别的文本",
    "is_final": false,  // 是否为最终结果
    "wav_name": ""     // 音频文件名（可选）
}
```

### HTTP 端点

- **健康检查**: `GET /health`
  - 返回服务状态和连接数

## 注意事项

1. 模型文件需要已经下载到 `models/` 目录
2. 音频格式：PCM 16-bit，16kHz，单声道
3. 服务使用 CPU 推理（ONNX Runtime CPUExecutionProvider）
4. ITN（逆文本正规化）功能暂未实现

## 开发

项目结构：

```
td-asr/
├── td_asr/
│   ├── __init__.py
│   ├── config.py          # 配置
│   ├── audio.py           # 音频处理
│   ├── frontend.py        # 特征提取
│   ├── models.py          # 模型加载和推理
│   ├── asr_engine.py      # 2-pass ASR 引擎
│   └── server.py          # FastAPI WebSocket 服务器
├── main.py                # 入口文件
├── models/                # 模型目录
└── pyproject.toml         # 项目配置

