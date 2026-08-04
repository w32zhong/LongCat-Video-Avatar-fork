# LongCat-Video Avatar 1.5：改动说明

## 安装 Conda 环境

```bash
conda create --name longcat-video-cu124 --channel conda-forge python=3.10.20 pip --yes
conda activate longcat-video-cu124

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'

python -m pip install --no-build-isolation -r requirements.txt
python -m pip install -r requirements_avatar.txt

python -m pip install \
  accelerate==1.14.0 \
  onnxruntime==1.23.2 \
  soundfile==0.14.0

conda install --override-channels --channel conda-forge --freeze-installed --yes \
  'ffmpeg=7.1.1=gpl_hbbdf940_911' \
  'x264=1!164.3095=h166bdaf_2'
```

关键版本：

| 包 | 版本 |
|---|---|
| Python | 3.10.20 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| flash_attn | 2.7.4.post1 |
| transformers | 4.41.0 |
| diffusers | 0.35.1 |
| accelerate | 1.14.0 |
| audio-separator | 0.30.2 |
| librosa | 0.11.0 |
| soundfile | 0.14.0 |
| numpy | 1.26.4 |
| scikit-image | 0.25.2 |
| pyloudnorm | 0.1.1 |
| chardet | 5.2.0 |
| onnx | 1.18.0 |
| onnxruntime | 1.23.2 |
| FFmpeg | 7.1.1 (`gpl_hbbdf940_911`) |
| x264 | 164.3095 (`h166bdaf_2`) |

## 下载模型

```bash
MODEL_STORE=/path/to/LongCat-Video-Avatar-1.5
mkdir -p "$MODEL_STORE"
cd "$MODEL_STORE"
```

```bash
bash <(curl -sSL https://g.bodaay.io/hfd) download \
  meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir .
```

```bash
bash <(curl -sSL https://g.bodaay.io/hfd) download \
  meituan-longcat/LongCat-Video \
  -F 'diffusion_pytorch_model.safetensors,spiece.model,tokenizer.json' \
  --local-dir .
```

```bash
bash <(curl -sSL https://g.bodaay.io/hfd) download \
  meituan-longcat/LongCat-Video \
  -F 'model-00001-of-00005,model-00002-of-00005,model-00003-of-00005,model-00004-of-00005,model-00005-of-00005' \
  --local-dir .
```

```bash
CHECKPOINT_DIR="$MODEL_STORE/meituan-longcat/LongCat-Video-Avatar-1.5"
```

`LongCat-Video` 必须与 `LongCat-Video-Avatar-1.5` 位于同一目录：

```text
meituan-longcat/
├── LongCat-Video/
└── LongCat-Video-Avatar-1.5/
```

## 生成 8 步 INT8 模型

模型下载完成后执行一次：

```bash
cd /path/to/LongCat-Video
conda activate longcat-video-cu124
python prepare_int8_dmd_model.py --checkpoint_dir "$CHECKPOINT_DIR"
```

生成 `$CHECKPOINT_DIR/base_model_int8_dmd_merged`，供四卡去噪使用。再次执行会检测已有结果并直接退出。

当前跑通的配置使用 FlashAttention 2；不需要修改模型配置。

## Quick Start

```bash
conda activate longcat-video-cu124
```

```bash
CHECKPOINT_DIR=/path/to/model-store/meituan-longcat/LongCat-Video-Avatar-1.5
INPUT_JSON=assets/avatar/input_example.json
OUTPUT_DIR=outputs_avatar_run
GPUS=0,1,3,4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

默认输出帧率为 25 FPS；脚本根据音频时长自动计算生成帧数。

复制并编辑通用配置文件 `assets/avatar/input_example.json`：

```json
{
  "prompt": "A cartoon livestream host speaks energetically to the camera, naturally moving the head, blinking, smiling, and making subtle upper-body gestures.",
  "cond_image": "/path/to/reference_image.png",
  "cond_audio": {
    "person1": "/path/to/speech.wav"
  }
}
```

1. 准备图片和音频条件：

此阶段只做预处理，不生成视频：

- 用四张 GPU 运行 UMT5 text encoder，将 prompt 编码后卸载 text encoder。
- 将图片裁剪到目标分辨率，用 VAE 编码为 `condition_latents`。
- 用 `Kim_Vocal_2.onnx` 分离人声，并重采样到 16 kHz。
- 用 Whisper-large-v3 提取按 25 FPS 对齐的音频 embedding，用于控制嘴型。
- 将图片 latent、音频 embedding 和输入信息保存到 `inputs.pt`。
- 保存后释放 UMT5、VAE 和 Whisper，再进入 DiT 去噪，避免这些模型同时占用显存导致 OOM。

```bash
CUDA_VISIBLE_DEVICES="$GPUS" python run_demo_avatar_single_low_vram.py prepare \
  --input_json "$INPUT_JSON" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --cache_path "$OUTPUT_DIR/inputs.pt"
```

默认使用真实 prompt。若要跳过 text encoder 并使用零 prompt embedding：

```bash
CUDA_VISIBLE_DEVICES=0 python run_demo_avatar_single_low_vram.py prepare \
  --input_json "$INPUT_JSON" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --cache_path "$OUTPUT_DIR/inputs.pt" \
  --skip_text_encoder
```

2. 四卡去噪：

此阶段真正生成视频内容，但输出仍是 latent，不是 MP4：

- 输入 `inputs.pt`：图片 `condition_latents`、Whisper 音频 embedding、prompt embedding 和目标宽高。
- 输入 `base_model_int8_dmd_merged`、随机噪声和固定 seed。
- 四张 GPU 通过 Context Parallel 分担视频 token 计算。
- INT8 DiT 执行 8 步蒸馏去噪；音频 embedding 控制嘴型和说话动作。
- DiT block 每组 n 层从 CPU 移到 GPU，计算完一组后再移回 CPU。
- 输出 `latent.pt`：生成视频的压缩表示，不能直接播放。

```bash
CUDA_VISIBLE_DEVICES="$GPUS" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  run_demo_avatar_single_low_vram.py denoise \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --cache_path "$OUTPUT_DIR/inputs.pt" \
  --latent_path "$OUTPUT_DIR/latent.pt" \
  --context_parallel_size 4 \
  --dit_subfolder base_model_int8_dmd_merged \
  --sequential_block_cpu_offload \
  --block_offload_group_size 32
```

3. 解码：

此阶段将 `latent.pt` 转换为可播放视频：

- 输入 `latent.pt`：四卡去噪生成的视频 latent。
- 输入 `inputs.pt`：读取原始音频路径、分辨率等信息。
- 输入 VAE 模型：将 latent 解码为 RGB 视频帧。
- 将视频帧编码为 25 FPS H.264，并裁切、合入原始音频作为 AAC 音轨。
- 输出 `ai2v_demo_1_low_vram.mp4`。

```bash
CUDA_VISIBLE_DEVICES=0 python run_demo_avatar_single_low_vram.py decode \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --cache_path "$OUTPUT_DIR/inputs.pt" \
  --latent_path "$OUTPUT_DIR/latent.pt" \
  --output_dir "$OUTPUT_DIR"
```

输出：`outputs_avatar_run/ai2v_demo_1_low_vram.mp4`

## 修改的代码

### `longcat_video/modules/quantization.py`

- 增加 meta-device INT8 模型初始化。
- 增加 safetensors 分片加载和 `assign=True` 参数分配。
- `load_quantized_dit()` 增加 `low_cpu_mem_usage` 和 `device`。
- 增加 `merge_lora_into_quantized_model()`，将 DMD LoRA 合入 INT8 权重并重新量化。

### `longcat_video/modules/avatar/longcat_video_dit_avatar.py`

- 增加 `enable_sequential_block_cpu_offload()`。
- 每次将 n 个 DiT block 移到 GPU，连续计算后再移回 CPU。
- 默认关闭，不影响原调用方式。

### `longcat_video/pipeline_longcat_video_avatar.py`

`generate_ai2v()` 增加：

- `prompt_embeds`
- `prompt_attention_mask`
- `condition_latents`

用于把 text encoder、VAE 编码与 DiT 去噪拆成不同阶段，避免同时占用显存。

### `run_demo_avatar_single_low_vram.py`

新增三阶段低显存入口：

- `prepare`：四卡 UMT5 prompt 编码、VAE 图片编码、人声分离、Whisper 音频编码。
- `denoise`：四卡 Context Parallel、INT8 DMD、8 步蒸馏、block offload。
- `decode`：单卡 tiled VAE 解码和音频封装。
- `prepare`、`decode` 使用 `torch.inference_mode()`。

### 其他文件

- 新增 `assets/avatar/input_example.json`：通用输入配置示例。
- 修改 `.gitignore`：忽略 `/audio_temp_file_*/` 和 `/outputs_avatar_*/`。
