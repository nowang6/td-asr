
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


