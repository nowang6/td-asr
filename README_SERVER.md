# TD-ASR 实时语音听写服务

基于 FastAPI 和 ONNX Runtime 实现的实时语音识别 WebSocket 服务，参考 FunASR C++ 官方实现。

## 功能特性

- **2-pass 识别**: 在线流式识别 + 离线高精度识别
- **VAD 语音活动检测**: 自动检测语音起止点
- **标点符号预测**: 自动添加标点符号
- **纯 ONNX Runtime**: 不依赖 FunASR，直接使用 ONNX 推理
- **WebSocket 协议**: 兼容 FunASR C++ 客户端协议

## 架构说明

参考 `cpp-implement/bin/funasr-wss-server-2pass.cpp` 的架构：

```
音频流 → VAD → Online ASR (流式结果) → Offline ASR (最终结果) → PUNC → 文本输出
```

### 核心模块

1. **frontend.py**: 特征提取 (Fbank + LFR)
   - VAD: lfr_m=5, lfr_n=1 → 400维特征
   - ASR: lfr_m=7, lfr_n=6 → 560维特征

2. **vad_model.py**: FSMN VAD 模型
   - 语音活动检测
   - 自动分段

3. **asr_model.py**: 在线/离线 ASR 模型
   - OnlineASRModel: 流式识别（带 encoder cache）
   - OfflineASRModel: 离线高精度识别

4. **punc_model.py**: 标点符号模型
   - CT-Transformer 标点预测

5. **asr_engine.py**: ASR 引擎
   - 整合所有组件
   - 2-pass 识别流程

6. **server.py**: FastAPI WebSocket 服务
   - WebSocket 协议处理
   - 每个连接独立的 ASR 引擎实例

## 安装

依赖已在 `pyproject.toml` 中定义：

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 下载模型

如果模型还未下载，可以使用下载脚本：

```bash
python download_models.py
```

该脚本会从 ModelScope 下载以下模型到 `models/` 目录：

- VAD: `damo/speech_fsmn_vad_zh-cn-16k-common-onnx`
- 在线 ASR: `damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx`
- 离线 ASR: `damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx`
- 标点: `damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727-onnx`

注意：模型下载需要安装 FunASR：
```bash
pip install funasr
```

## 启动服务

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

或使用默认参数：

```bash
python main.py
```

## WebSocket 协议

### 连接地址

```
ws://localhost:10095/ws
```

### 消息格式

#### 1. 客户端发送配置消息（JSON）

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

#### 2. 客户端发送音频数据（二进制）

- 格式: PCM 16-bit
- 采样率: 16000 Hz
- 声道: 单声道
- 建议块大小: 800 samples (50ms)

#### 3. 服务器返回识别结果（JSON）

```json
{
    "text": "识别的文本",
    "is_final": false,
    "timestamp": 0,
    "mode": "2pass-online"
}
```

- `is_final=false`: 在线流式结果（实时）
- `is_final=true`: 离线最终结果（高精度）

#### 4. 客户端发送结束消息（JSON）

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

## 已知问题

所有已知问题及解决方案详见 `ISSUES.md`：

1. ✅ CMVN 文件格式（Kaldi Nnet 格式）
2. ✅ VAD 和 ASR 特征维度不匹配
3. ✅ Encoder 缓存输入
4. ✅ WebSocket 协议
5. ✅ 每个连接独立的 ASR 引擎实例

## 测试

可以使用 C++ 客户端测试：

```bash
cd cpp-implement/build
./funasr-wss-client-2pass \
    --server-ip 127.0.0.1 \
    --port 10095 \
    --wav-path /path/to/audio.wav
```

## 项目结构

```
td-asr/
├── src/
│   ├── __init__.py
│   ├── utils.py           # 工具函数
│   ├── frontend.py        # 前端特征提取
│   ├── vad_model.py       # VAD 模型
│   ├── asr_model.py       # ASR 模型
│   ├── punc_model.py      # 标点模型
│   ├── asr_engine.py      # ASR 引擎
│   └── server.py          # FastAPI 服务
├── models/                # 模型文件
├── main.py               # 主入口
├── download_models.py    # 模型下载脚本
├── pyproject.toml        # 项目配置
└── ISSUES.md            # 已知问题
```

## 参考

- FunASR C++ 实现: `cpp-implement/bin/funasr-wss-server-2pass.cpp`
- 问题记录: `ISSUES.md`

## 许可证

MIT License

