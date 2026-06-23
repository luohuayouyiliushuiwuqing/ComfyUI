# ComfyUI 工作流配置指南

## 📁 工作流文件

| 文件 | 用途 | 模型组合 |
|------|------|----------|
| `FHDR_workflow_Default.json` | FHDR 文生图 | FHDR Q8 + T5-XXL |
| `Flux2_Klein_workflow_Default.json` | Flux2-Klein 文生图 | Flux2-Klein-9B + Qwen3-8B |

---

## 🔧 工作流 1: FHDR 文生图

### 模型配置

| 组件 | 模型文件 | 来源 | 大小 |
|------|----------|------|------|
| **DiT** | `FHDR_ComfyUI-Q8_0.gguf` | `text2image/` | 12GB |
| **Text Encoder** | `t5xxl_fp16.safetensors` | `text_encoders/` | 9.3GB |
| **VAE** | `ae.safetensors` | `vae/` | 320MB |

### 节点配置

```
UnetLoaderGGUF → FHDR_ComfyUI-Q8_0.gguf
CLIPLoaderGGUF → t5xxl_fp16.safetensors (type=flux2)
VAELoader      → ae.safetensors
KSampler       → euler, simple, 20 steps, cfg=1.0
VAEDecode      → SaveImage
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 1024×1024 | Flux 推荐分辨率 |
| Steps | 20 | 采样步数 |
| CFG | 1.0 | 引导强度 (Flux 推荐低 CFG) |
| Sampler | euler | 采样器 |
| Scheduler | simple | 调度器 |

### 模型来源

- **FHDR_ComfyUI-Q8_0.gguf**: Flux 架构的 GGUF 量化模型，12B 参数
- **t5xxl_fp16.safetensors**: T5-XXL 文本编码器，输出 4096 维特征
- **ae.safetensors**: Flux 标准 VAE

---

## 🔧 工作流 2: Flux2-Klein 文生图

### 模型配置

| 组件 | 模型文件 | 来源 | 大小 |
|------|----------|------|------|
| **DiT** | `Flux2-Klein-9B-True-v2-Q6_K.gguf` | `flux2/OfficialModel/` | 7.4GB |
| **Text Encoder** | `qwen3_text_encoder.safetensors` | `text_encoders/` | 16GB |
| **VAE** | `flux2-vae.safetensors` | `flux2/OfficialModel/` | 321MB |

### 节点配置

```
UnetLoaderGGUF → Flux2-Klein-9B-True-v2-Q6_K.gguf
CLIPLoaderGGUF → qwen3_text_encoder.safetensors (type=flux2)
VAELoader      → flux2-vae.safetensors
KSampler       → dpmpp_2m, karras, 28 steps, cfg=1.5
VAEDecodeTiled → SaveImage
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 1024×1024 | Flux 推荐分辨率 |
| Steps | 28 | 采样步数 (比 FHDR 多) |
| CFG | 1.5 | 引导强度 (略高于 FHDR) |
| Sampler | dpmpp_2m | 高质量采样器 |
| Scheduler | karras | 非线性调度器 |

### 模型来源

- **Flux2-Klein-9B-True-v2-Q6_K.gguf**: Flux2 架构的 GGUF 量化模型，9B 参数，无审查版本
- **qwen3_text_encoder.safetensors**: Qwen3-8B 文本编码器，输出 12288 维特征 https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder/blob/main/model.safetensors
- **flux2-vae.safetensors**: Flux2 专用 VAE

---

## ⚠️ 关键区别

### 1. 文本编码器维度

| 工作流 | Text Encoder | 输出维度 | 兼容性 |
|--------|--------------|----------|--------|
| FHDR | T5-XXL | 4096 | 只能配 FHDR |
| Flux2-Klein | Qwen3-8B | 12288 | 只能配 Flux2-Klein |

### 2. CLIPLoaderGGUF type 参数

| 工作流 | type 值 | 说明 |
|--------|---------|------|
| FHDR | `flux2` | Flux 架构 |
| Flux2-Klein | `flux2` | Flux2 架构 |

**注意**: 两个工作流的 type 都选 `flux2`，但加载的编码器不同！

### 3. 采样参数

| 参数 | FHDR | Flux2-Klein |
|------|------|-------------|
| Steps | 20 | 28 |
| CFG | 1.0 | 1.5 |
| Sampler | euler | dpmpp_2m |
| Scheduler | simple | karras |

### 4. VAE 解码

| 工作流 | VAE 节点 | VAE 文件 |
|--------|----------|----------|
| FHDR | VAEDecode | ae.safetensors |
| Flux2-Klein | VAEDecodeTiled | flux2-vae.safetensors |

---

## 📦 模型安装路径

```
ComfyUI/
├── models/
│   ├── unet/
│   │   ├── FHDR_ComfyUI-Q8_0.gguf              ← FHDR DiT
│   │   └── Flux2-Klein-9B-True-v2-Q6_K.gguf    ← Flux2 DiT
│   ├── text_encoders/
│   │   ├── t5xxl_fp16.safetensors               ← FHDR 编码器
│   │   └── qwen3_text_encoder.safetensors        ← Flux2 编码器
│   └── vae/
│       ├── ae.safetensors                        ← FHDR VAE
│       └── flux2-vae.safetensors                 ← Flux2 VAE
```

---

## 🎯 选择建议

| 需求 | 推荐工作流 | 原因 |
|------|-----------|------|
| 通用图像生成 | FHDR | 稳定、质量高、有审查 |
| 无审查内容 | Flux2-Klein | 无内容限制 |
| 快速测试 | FHDR | 步数少、速度快 |
| 高质量输出 | Flux2-Klein | 更多步数、更好采样器 |

---

## ❌ 常见错误

### 错误 1: Flux2-Klein + T5-XXL

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x4096 and 12288x4096)
```

**原因**: Flux2-Klein 需要 Qwen3-8B (12288维)，不能用 T5-XXL (4096维)

### 错误 2: 使用错误的 VAE

```
VAE 解码失败或 OOM
```

**原因**: Flux2-Klein 应使用 flux2-vae.safetensors

### 错误 3: CLIPLoaderGGUF type 选择错误

**解决**: 两个工作流的 type 都选 `flux2`

---

**最后更新**: 2026-06-23  
**维护者**: MiMo Code Agent