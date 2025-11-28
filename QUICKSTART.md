# TD-ASR 快速启动指南

## 1. 检查模型

首先检查所有模型是否已下载：

```bash
python check_models.py
```

如果输出显示所有模型都 Ready，则可以直接启动服务。

## 2. 启动服务

### 方式 1：使用 Python 直接启动

```bash
python main.py
```

### 方式 2：使用 Shell 脚本启动

```bash
chmod +x run_server.sh
./run_server.sh
```

### 方式 3：自定义参数启动

```bash
python main.py \
    --host 0.0.0.0 \
    --port 10095 \
    --vad-dir models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx \
    --online-model-dir models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx \
    --offline-model-dir models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx \
    --punc-dir models/damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx \
    --quantize true
```

服务启动后会监听在 `0.0.0.0:10095`，可以通过 WebSocket 连接 `ws://localhost:10095/ws`

## 3. 测试服务

### 方式 1：使用 Python 测试客户端

```bash
python test_client.py --audio /path/to/your/audio.wav
```

### 方式 2：使用 C++ 官方客户端

如果你已经编译了 C++ 客户端：

```bash
cd cpp-implement/build
./funasr-wss-client-2pass \
    --server-ip 127.0.0.1 \
    --port 10095 \
    --wav-path /path/to/audio.wav
```

### 方式 3：使用 WebSocket 测试工具

你可以使用任何 WebSocket 客户端工具，按照以下协议发送数据：

#### 步骤 1：连接 WebSocket

```
ws://localhost:10095/ws
```

#### 步骤 2：发送配置消息（JSON）

```json
{
    "mode": "2pass",
    "chunk_size": [5, 10, 5],
    "wav_name": "test.wav",
    "wav_format": "pcm",
    "audio_fs": 16000,
    "is_speaking": true
}
```

#### 步骤 3：发送音频数据（二进制）

发送 PCM 16-bit 格式的音频数据，建议每次发送 800 samples (1600 bytes = 50ms @ 16kHz)

#### 步骤 4：接收识别结果（JSON）

服务器会实时返回识别结果：

**在线结果（流式）**:
```json
{
    "text": "识别的文本",
    "is_final": false,
    "timestamp": 0,
    "mode": "2pass-online"
}
```

**离线结果（最终）**:
```json
{
    "text": "最终识别的文本，带标点符号",
    "is_final": true,
    "timestamp": 0,
    "mode": "2pass-offline"
}
```

#### 步骤 5：发送结束消息（JSON）

```json
{
    "is_speaking": false
}
```

或

```json
{
    "is_finished": true
}
```

## 4. 性能优化建议

### CPU 优化

如果你的 CPU 支持，可以设置环境变量来启用优化：

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python main.py
```

### 内存优化

如果内存有限，可以只使用在线模型或离线模型：

```bash
# 只使用在线模型（快速但精度稍低）
python main.py --offline-model-dir ""

# 只使用离线模型（精度高但延迟较高）
python main.py --online-model-dir ""
```

## 5. 常见问题

### Q: 模型加载失败？

A: 确保所有模型文件都已下载。运行 `python check_models.py` 检查。

### Q: 识别结果是空的？

A: 检查：
1. 音频格式是否正确（PCM 16-bit, 16kHz, 单声道）
2. 是否正确发送了配置消息
3. 查看服务器日志输出

### Q: 连接断开？

A: 检查：
1. 网络连接是否正常
2. 服务器是否正在运行
3. 是否正确发送了结束消息

### Q: 识别速度慢？

A: 尝试：
1. 使用量化模型（`--quantize true`，默认）
2. 调整线程数（修改代码中的 `inter_op_num_threads` 和 `intra_op_num_threads`）
3. 只使用在线模型

## 6. 架构说明

```
WebSocket 连接
    ↓
配置消息（JSON）
    ↓
音频流（Binary PCM）
    ↓
特征提取（Fbank + LFR）
    ↓
┌─────────────────────┐
│   2-Pass 识别        │
│                     │
│  ┌──────────────┐   │
│  │ Online ASR   │ ──┼── 流式结果（is_final=false）
│  │  (实时)      │   │
│  └──────────────┘   │
│                     │
│  ┌──────────────┐   │
│  │ Offline ASR  │ ──┼── 最终结果（is_final=true）
│  │  (高精度)    │   │
│  └──────────────┘   │
│        ↓            │
│  ┌──────────────┐   │
│  │ Punctuation  │   │
│  │  (标点)      │   │
│  └──────────────┘   │
└─────────────────────┘
    ↓
识别结果（JSON）
```

## 7. 参考文档

- 完整文档：`README_SERVER.md`
- 已知问题：`ISSUES.md`
- C++ 参考实现：`cpp-implement/bin/funasr-wss-server-2pass.cpp`

