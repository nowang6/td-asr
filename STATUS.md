# TD-ASR 实现状态

## ✅ 已完成

### 核心功能
- [x] FastAPI WebSocket 服务器
- [x] 前端特征提取（Fbank + LFR）
- [x] CMVN 加载和应用（Kaldi Nnet 格式）
- [x] VAD 模型集成
- [x] 在线 ASR 模型（带 encoder cache）
- [x] 离线 ASR 模型
- [x] 标点符号模型（基础实现）
- [x] 2-pass 识别流程

### 协议兼容
- [x] 兼容 C++ 客户端协议
- [x] WebSocket 端点：`/` 和 `/ws`
- [x] 支持 JSON 配置消息
- [x] 支持二进制 PCM 音频数据
- [x] 按 800 samples chunk 处理

### 工具脚本
- [x] `main.py` - 服务器启动脚本
- [x] `test_client.py` - Python 测试客户端
- [x] `check_models.py` - 模型检查工具
- [x] `test_models.py` - 模型输入输出检查
- [x] `test_asr_simple.py` - ASR 模型简单测试

## 🔧 最近修复

### 2025-11-28 修复记录

1. **CMVN 加载修复**
   - 正确解析 `[` `]` 之间的数值
   - 维度检查：VAD 400维，ASR 560维

2. **WebSocket 路径修复**
   - 添加根路径 `/` 支持（兼容 C++ 客户端）
   - 保留 `/ws` 路径

3. **Decoder 输出处理**
   - 优先使用 `sample_ids` 输出（直接的 token IDs）
   - Fallback 到 `logits` + argmax

4. **流式识别修复**
   - 每个 800 samples chunk 立即处理
   - 维护 encoder cache 状态
   - 匹配 C++ 实现的处理逻辑

5. **标点符号简化**
   - 禁用错误的规则式标点
   - 仅在句末添加句号

## 🐛 已知问题

### 1. 在线识别可能为空
**症状**：在线 ASR 返回空字符串或只有特殊 token

**可能原因**：
- Encoder 输出维度可能不匹配 decoder 期望
- Decoder cache 初始化可能有误
- Token 转换可能过滤掉了所有 token

**调试方法**：
```bash
# 查看详细日志
python main.py  # 会输出 DEBUG 日志

# 关键日志：
# - Online encoder output: shape=(T, 512)
# - Online decoder tokens: N tokens, first 20: [...]
# - Converted N tokens to text: '...'
```

### 2. 标点符号功能未完整实现
**现状**：只在句末添加句号

**TODO**：实现完整的 PUNC 模型 ONNX 推理

### 3. 无 ITN 支持
**现状**：未实现 ITN（数字转文本）

**TODO**：集成 FST ITN 模型

## 🚀 测试方法

### 启动服务器
```bash
python main.py
```

### 使用 C++ 客户端测试
```bash
cd cpp-implement/build
./funasr-wss-client-2pass \
    --server-ip 127.0.0.1 \
    --port 10095 \
    --wav-path data/test.wav \
    --is-ssl 0
```

### 使用 Python 客户端测试
```bash
python test_client.py --audio test.wav
```

## 📊 性能对比

### C++ 服务器
- 在线识别：实时流式输出（每 800 samples）
- 离线识别：完整结果 + 标点 + 时间戳
- 标点符号：完整 PUNC 模型

### Python 服务器（当前）
- 在线识别：处理中（可能输出空）
- 离线识别：完整结果 + 基础标点
- 标点符号：基础实现（仅句末句号）

## 🔍 调试技巧

### 1. 检查模型输入输出
```bash
python test_models.py
```

### 2. 测试 ASR 模型基础功能
```bash
python test_asr_simple.py
```

### 3. 查看详细日志
服务器会输出：
- CMVN 维度
- Encoder/Decoder 输入输出名称
- 每个推理步骤的 shape
- Token 转换过程

### 4. 常见错误排查

**错误：`operands could not be broadcast together`**
- 原因：特征维度与 CMVN 不匹配
- 检查：CMVN shape 应该是 `(2, feature_dim)`

**错误：`Required inputs [...] are missing`**
- 原因：Decoder 需要的输入未提供
- 检查：是否正确提供了 `acoustic_embeds` 和所有 cache

**连接被拒绝（403）**
- 原因：WebSocket 路径不匹配
- 已修复：现在支持 `/` 和 `/ws` 两个路径

## 📝 下一步计划

1. **修复在线识别**
   - 调试 encoder/decoder 输出
   - 确认 cache 正确传递
   - 验证 token 转换

2. **实现完整标点**
   - 加载 PUNC 模型的 tokens
   - 实现 ONNX 推理
   - 添加标点预测逻辑

3. **添加 ITN 支持**
   - 集成 FST ITN
   - 处理数字、日期等

4. **性能优化**
   - 支持 GPU 推理
   - 批处理优化
   - 减少内存占用

## 📚 参考

- C++ 实现：`cpp-implement/bin/funasr-wss-server-2pass.cpp`
- 已知问题：`ISSUES.md`
- 快速开始：`QUICKSTART.md`
- 项目总结：`PROJECT_SUMMARY.md`

---

**更新时间**: 2025-11-28 16:50
**测试状态**: 可连接，离线识别正常，在线识别调试中

