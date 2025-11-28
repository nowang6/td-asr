
## 1. CMVN 文件格式问题

**问题描述**：
- 初始实现假设 CMVN 文件是纯数值格式（每行包含数值）
- 实际 CMVN 文件是 Kaldi Nnet 格式，包含 `<Nnet>`, `<AddShift>`, `<Rescale>`, `<LearnRateCoef>` 等标签

**错误信息**：
```
ValueError: could not convert string to float: '<Nnet>'
```

**解决方案**：
- 修改 `_load_cmvn()` 方法，解析 Kaldi Nnet 格式
- 从 `<AddShift>` 后的 `<LearnRateCoef>` 行提取 mean 值（索引 3 到倒数第二个）
- 从 `<Rescale>` 后的 `<LearnRateCoef>` 行提取 std 值（索引 3 到倒数第二个）
- 返回格式：`[mean, var]` 的 numpy 数组

**参考**：`cpp-implement/third_party/onnxruntime/src/fsmn-vad.cpp` 中的 `LoadCmvn()` 实现

---


## 3. VAD 和 ASR 特征维度不匹配

**问题描述**：
- VAD 模型需要 400 维特征（lfr_m=5, lfr_n=1，80*5=400）
- ASR 模型需要 560 维特征（lfr_m=7, lfr_n=6，80*7=560）
- 初始实现使用同一个前端和 CMVN，导致维度不匹配

**错误信息**：
```
ValueError: operands could not be broadcast together with shapes (82,560) (400,)
```

**解决方案**：
- 为 VAD 创建单独的前端：`WavFrontendOnline(lfr_m=5, lfr_n=1)`
- 为 ASR 保留原前端：`WavFrontendOnline(lfr_m=7, lfr_n=6)`
- 分别提取 VAD 特征和 ASR 特征
- 分别应用各自的 CMVN 统计信息

**参考**：
- VAD 配置：`models/damo/speech_fsmn_vad_zh-cn-16k-common-onnx/config.yaml`
- ASR 配置：`models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx/config.yaml`

---

## 4. Encoder 缓存输入缺失问题

**问题描述**：
- Online ASR 模型的 encoder 在运行时需要缓存输入（`in_cache0`, `in_cache1`, `in_cache2`, `in_cache3`）
- 但 ONNX 模型的输入定义中只显示 `['speech', 'speech_lengths']`
- 缓存输入在模型定义中未显式声明，但运行时必需

**错误信息**：
```
Required inputs (['in_cache0', 'in_cache1', 'in_cache2', 'in_cache3']) are missing from input feed (['speech']).
```

**解决方案**：
- 在 `_init_encoder_cache()` 中，如果模型定义中没有缓存输入，自动创建默认缓存
- 默认创建 4 个缓存：`in_cache0`, `in_cache1`, `in_cache2`, `in_cache3`
- 默认形状：`[1, 512, 10]`（batch_size, hidden_dim, cache_len）
- 在 `infer()` 方法中，如果检测不到缓存输入名称，使用默认创建的缓存名称

**注意**：这是 Paraformer 在线模型的特殊要求，缓存用于维护流式推理的状态

---

## 5. WebSocket 协议不匹配

**问题描述**：
- 初始实现使用简化的协议（直接发送音频 + 控制消息）
- C++ 客户端使用更复杂的协议：
  1. 首先发送 JSON 配置消息（包含 `mode`, `chunk_size`, `wav_name`, `wav_format`, `audio_fs`, `is_speaking` 等）
  2. 然后发送二进制音频数据（PCM 16-bit）
  3. 最后发送结束消息（`is_speaking=false` 或 `is_finished=true`）

**解决方案**：
- 更新服务器端以正确处理初始 JSON 配置消息
- 累积音频数据到缓冲区
- 按 800 样本（50ms）的块处理音频，与 C++ 实现一致
- 当收到 `is_speaking=false` 时，处理剩余缓冲区并发送最终结果

**参考**：`cpp-implement/bin/funasr-wss-client-2pass.cpp` 和 `cpp-implement/bin/websocket-server-2pass.cpp`

---

## 6. 模型文件命名约定

**问题描述**：
- 需要正确识别模型文件名称
- Online ASR 模型有两个文件：`model_quant.onnx`（encoder）和 `decoder_quant.onnx`（decoder）
- 其他模型只有一个文件：`model_quant.onnx`

**解决方案**：
- 在 `OnlineASRModel` 中分别加载 encoder 和 decoder
- 使用 `_load_onnx_model("model")` 加载 encoder
- 使用 `_load_onnx_model("decoder")` 加载 decoder

---

## 7. 音频格式处理

**问题描述**：
- 客户端发送的是 PCM 16-bit 格式的二进制数据
- 需要正确转换为 float32 数组用于模型推理

**解决方案**：
- 实现 `bytes_to_float32()` 函数
- 将 int16 转换为 float32 并归一化到 [-1, 1] 范围

---

## 8. 每个连接需要独立的 ASR 引擎实例

**问题描述**：
- 初始实现使用全局共享的 ASR 引擎
- 多个连接会共享状态，导致识别结果混乱

**解决方案**：
- 为每个 WebSocket 连接创建独立的 ASR 引擎实例
- 使用工厂模式管理引擎创建（共享模型路径配置）

---

## 9. WebSocket 路径兼容性问题

**问题描述**：
- C++ 客户端连接到根路径 `/`
- 初始实现只在 `/ws` 路径提供 WebSocket 服务
- 导致 403 Forbidden 错误

**错误信息**：
```
[error] Server handshake response error: websocketpp.processor:20 (Invalid HTTP status.)
INFO: 127.0.0.1:54838 - "WebSocket /" 403
```

**解决方案**：
- 在 FastAPI 中同时注册 `/` 和 `/ws` 两个 WebSocket 端点
- 两个端点共享同一个处理函数 `websocket_handler`

**代码**：
```python
@app.websocket("/")
async def websocket_endpoint_root(websocket: WebSocket):
    await websocket_handler(websocket)

@app.websocket("/ws")
async def websocket_endpoint_ws(websocket: WebSocket):
    await websocket_handler(websocket)
```

**参考**：`cpp-implement/bin/funasr-wss-client-2pass.cpp` 连接到根路径

---

## 10. 前端流式特征提取问题

**问题描述**：
- 单个 800 samples chunk (50ms) 太小，无法提取 LFR 特征
- 在线 ASR 的 LFR 参数：`lfr_m=7, lfr_n=6`（需要至少 7 个连续 fbank 帧）
- 800 samples 只能提取 3-4 个 fbank 帧，不足以生成 LFR 帧

**错误现象**：
```
Extracted features: shape=empty
No features extracted, skipping inference
```

**错误方案**：
- ❌ 累积所有音频后再提取特征（会导致延迟过高）
- ❌ 每个 chunk 单独提取特征（chunk 太小，无法提取）

**正确解决方案**：
- 使用 Frontend 的 `extract_feat_streaming()` 方法
- 该方法内部维护音频缓冲区，累积到足够长度才提取特征
- 提取后保留剩余样本供下次使用

**代码**：
```python
# 添加 chunk 到 frontend 内部缓冲区
self.online_asr_model.frontend.add_audio(audio_chunk)

# 提取流式特征（内部维护状态）
features, consumed = self.online_asr_model.frontend.extract_feat_streaming(return_samples=True)
```

**参考**：C++ 实现中 `FunTpassInferBuffer` 内部维护前端缓冲区

---

## 11. Decoder Cache 持久化问题

**问题描述**：
- 在线模型需要 16 个 decoder cache (`in_cache_0` ~ `in_cache_15`)
- 初始实现每次调用都使用零初始化的 cache
- Decoder 无法利用历史信息，每次都从头开始解码
- 导致识别结果不连贯或输出只有结束符号 `[2]` (</s>)

**错误现象**：
```
Online decoder tokens: 1 tokens, first 20: [2]
Converted 1 tokens to text: ''
```

**解决方案**：
1. 在 `OnlineASRModel` 中维护持久的 `self.decoder_cache`
2. 初始化时创建 decoder cache：
```python
def _init_decoder_cache(self) -> Dict[str, np.ndarray]:
    cache = {}
    for i in range(16):
        cache_name = f'in_cache_{i}'
        if cache_name in self.decoder_input_names:
            # Get shape from model
            for inp in self.decoder_session.get_inputs():
                if inp.name == cache_name:
                    shape = [1 if (isinstance(s, str) or s < 0) else s for s in shape]
                    cache[cache_name] = np.zeros(shape, dtype=np.float32)
    return cache
```

3. 每次 decode 后更新 cache：
```python
# Decode 后
for i, name in enumerate(self.decoder_output_names):
    if name.startswith('out_cache_'):
        input_cache_name = name.replace('out_cache_', 'in_cache_')
        self.decoder_cache[input_cache_name] = outputs[i]
```

4. 下次 decode 时使用更新后的 cache：
```python
# 添加持久化的 decoder cache
for cache_name, cache_value in self.decoder_cache.items():
    if cache_name in self.decoder_input_names:
        inputs[cache_name] = cache_value
```

**参考**：在线模型的流式解码需要维护历史状态

---

## 12. 在线识别内容不准确（调试中）

**问题描述**：
- 在线识别有输出，但内容不正确
- 实际输出："有"、"的"、"要"
- 期望输出："张三丰"、"你好"、"啊请向"

**当前状态**：
- ✅ 流式识别 pipeline 工作
- ✅ Decoder 有输出（不只是结束符号）
- ❌ 输出内容不对

**可能原因**：
1. **Acoustic embeds 问题**：当前使用零矩阵，可能需要实际的 acoustic embedding
2. **Decoder cache 更新问题**：虽然更新了，但可能维度或内容有误
3. **Encoder 输出问题**：Encoder 没有维护 cache（该模型可能不需要 encoder cache）

**调试方法**：
- 查看完整的 decoder cache shape
- 对比 C++ 和 Python 的 encoder 输出
- 检查 acoustic_embeds 是否需要非零值

**临时状态**：部分工作，需要进一步调试

---

## 13. 标点符号位置不正确

**问题描述**：
- 规则式标点实现会在错误位置插入逗号
- 实际输出："张三丰你，好啊请向，前看不要"
- 期望输出："张三丰，你好啊，请向前看"

**错误方案**：
```python
# 每 4 个字符添加逗号（错误）
if (i + 1) % 4 == 0:
    result.append('，')
```

**临时解决方案**：
- 禁用规则式标点
- 仅在句末添加句号

**最终解决方案（TODO）**：
- 实现完整的 PUNC 模型 ONNX 推理
- 加载 PUNC 模型的 tokens
- 正确预测标点位置


