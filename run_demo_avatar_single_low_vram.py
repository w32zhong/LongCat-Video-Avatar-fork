import argparse
import datetime
import gc
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from audio_separator.separator import Separator
from diffusers.utils import load_image

from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from longcat_video.context_parallel import context_parallel_util
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline, retrieve_latents


def _resolve_input_path(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return str(candidate.resolve())


def _clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_case(input_json: str):
    with open(input_json, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    image_path = _resolve_input_path(case["cond_image"])
    audio_path = _resolve_input_path(case["cond_audio"]["person1"])
    return case, image_path, audio_path


def _num_frames_for_audio(audio_duration: float, fps: int) -> int:
    required_frames = max(1, math.ceil(audio_duration * fps))
    return math.ceil((required_frames - 1) / 4) * 4 + 1


@torch.inference_mode()
def prepare(args):
    device = torch.device(args.device)
    case, image_path, audio_path = _load_case(args.input_json)
    checkpoint_dir = Path(args.checkpoint_dir)
    foundation_dir = checkpoint_dir.parent / "LongCat-Video"
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    vae = AutoencoderKLWan.from_pretrained(
        str(foundation_dir),
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae.enable_tiling()
    vae = vae.to(device)
    vae_pipe = LongCatVideoAvatarPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=None,
        dit=None,
        model_type="avatar-v1.5",
    )
    cp_split_hw = context_parallel_util.get_optimal_split(args.context_parallel_size)
    scale_factor_spatial = vae_pipe.vae_scale_factor_spatial * 2 * max(cp_split_hw)
    height, width = vae_pipe.get_condition_shape(
        image,
        args.resolution,
        scale_factor_spatial=scale_factor_spatial,
    )
    image_tensor = vae_pipe.video_processor.preprocess(
        image,
        height=height,
        width=width,
        resize_mode="crop",
    ).to(device=device, dtype=torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    condition_latents = retrieve_latents(
        vae.encode(image_tensor.unsqueeze(2)),
        generator,
        sample_mode="argmax",
    )
    condition_latents = vae_pipe.normalize_latents(condition_latents).cpu()
    del image_tensor, vae_pipe, vae
    _clear_cuda()
    print(f"[prepare] image latent ready: {tuple(condition_latents.shape)}, target={height}x{width}", flush=True)

    separator_dir = Path(args.work_dir) / "vocals"
    separator_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(
        output_dir=separator_dir,
        output_single_stem="vocals",
        model_file_dir=str(checkpoint_dir / "vocal_separator"),
    )
    separator.load_model("Kim_Vocal_2.onnx")
    separated = separator.separate(audio_path)
    if not separated:
        raise RuntimeError("Vocal separator returned no output")
    vocal_path = separator_dir / separated[0]
    if not vocal_path.exists():
        raise FileNotFoundError(f"Separated vocal not found: {vocal_path}")

    speech_array, sample_rate = librosa.load(str(vocal_path), sr=16000)
    audio_duration = len(speech_array) / sample_rate
    num_frames = _num_frames_for_audio(audio_duration, args.fps)
    generate_duration = num_frames / args.fps
    padding = math.ceil((generate_duration - len(speech_array) / sample_rate) * sample_rate)
    if padding > 0:
        speech_array = np.pad(speech_array, (0, padding))

    audio_checkpoint = checkpoint_dir / "whisper-large-v3"
    audio_encoder = get_audio_encoder(str(audio_checkpoint), "avatar-v1.5").to(device)
    audio_feature_extractor = get_audio_feature_extractor(str(audio_checkpoint), "avatar-v1.5")
    audio_pipe = LongCatVideoAvatarPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        scheduler=None,
        dit=None,
        audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor,
        model_type="avatar-v1.5",
    )
    full_audio_emb = audio_pipe.get_audio_embedding(
        speech_array,
        fps=args.fps,
        device=device,
        sample_rate=sample_rate,
        model_type="avatar-v1.5",
    )
    indices = torch.arange(5, device=full_audio_emb.device) - 2
    centers = torch.arange(num_frames, device=full_audio_emb.device).unsqueeze(1) + indices.unsqueeze(0)
    centers = centers.clamp(min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[centers][None].cpu()
    del full_audio_emb, audio_pipe, audio_encoder
    _clear_cuda()
    print(f"[prepare] audio embedding ready: {tuple(audio_emb.shape)}", flush=True)

    prompt_embeds = torch.zeros(1, 1, 512, 4096, dtype=torch.bfloat16)
    prompt_attention_mask = torch.zeros(1, 512, dtype=torch.int64)
    torch.save(
        {
            "condition_latents": condition_latents,
            "audio_emb": audio_emb,
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "height": height,
            "width": width,
            "image_path": image_path,
            "audio_path": audio_path,
            "prompt": case.get("prompt", ""),
            "num_frames": num_frames,
            "fps": args.fps,
        },
        cache_path,
    )
    print(f"[prepare] cache saved: {cache_path}", flush=True)


class _ConfigOnlyTextEncoder:
    dtype = torch.bfloat16
    config = SimpleNamespace(d_model=4096)


class _ConfigOnlyVAE:
    dtype = torch.bfloat16
    config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8)


def denoise(args):
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=24))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    context_parallel_util.init_context_parallel(
        context_parallel_size=args.context_parallel_size,
        global_rank=rank,
        world_size=world_size,
    )
    cp_split_hw = context_parallel_util.get_optimal_split(args.context_parallel_size)

    checkpoint_dir = Path(args.checkpoint_dir)
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=True)
    num_frames = int(cache["num_frames"])
    for loading_rank in range(world_size):
        if rank == loading_rank:
            print(f"[rank {rank}] loading merged INT8 DiT directly to {device}", flush=True)
            dit = load_quantized_dit(
                str(checkpoint_dir),
                subfolder=args.dit_subfolder,
                low_cpu_mem_usage=True,
                device="cpu" if args.sequential_block_cpu_offload else str(device),
                cp_split_hw=cp_split_hw,
            )
            if args.sequential_block_cpu_offload:
                dit.enable_sequential_block_cpu_offload(device)
        dist.barrier()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(checkpoint_dir),
        subfolder="scheduler",
        torch_dtype=torch.bfloat16,
    )
    pipe = LongCatVideoAvatarPipeline(
        tokenizer=None,
        text_encoder=_ConfigOnlyTextEncoder(),
        vae=_ConfigOnlyVAE(),
        scheduler=scheduler,
        dit=dit,
        audio_encoder=None,
        audio_feature_extractor=None,
        model_type="avatar-v1.5",
    )
    pipe.device = device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = load_image(cache["image_path"])

    latent = pipe.generate_ai2v(
        image=image,
        prompt=cache.get("prompt") or "A person speaking naturally.",
        negative_prompt=None,
        resolution=args.resolution,
        num_frames=num_frames,
        num_inference_steps=8,
        use_distill=True,
        text_guidance_scale=1.0,
        audio_guidance_scale=1.0,
        generator=generator,
        output_type="latent",
        audio_emb=cache["audio_emb"],
        prompt_embeds=cache["prompt_embeds"],
        prompt_attention_mask=cache["prompt_attention_mask"],
        condition_latents=cache["condition_latents"],
    )
    if rank == 0:
        output_path = Path(args.latent_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent.cpu(), output_path)
        print(f"[denoise] latent saved: {output_path}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


@torch.inference_mode()
def decode(args):
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    foundation_dir = checkpoint_dir.parent / "LongCat-Video"
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=True)
    latents = torch.load(args.latent_path, map_location="cpu", weights_only=True)

    vae = AutoencoderKLWan.from_pretrained(
        str(foundation_dir),
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    vae.enable_tiling()
    vae = vae.to(device)
    pipe = LongCatVideoAvatarPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        scheduler=None,
        dit=None,
        model_type="avatar-v1.5",
    )
    latents = pipe.denormalize_latents(latents.to(device=device, dtype=vae.dtype))
    output = vae.decode(latents, return_dict=False)[0]
    output = pipe.video_processor.postprocess_video(output)[0]
    frames = torch.from_numpy((output * 255).round().clip(0, 255).astype(np.uint8))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "ai2v_demo_1_low_vram"
    save_video_ffmpeg(frames, str(output_prefix), cache["audio_path"], fps=int(cache["fps"]), quality=5)
    print(f"[decode] video saved with prefix: {output_prefix}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Low-VRAM LongCat Avatar 1.5 AI2V runner")
    parser.add_argument("mode", choices=["prepare", "denoise", "decode"])
    parser.add_argument("--input_json", default="assets/avatar/single_example_1.json")
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--cache_path", default="./outputs_avatar_single/low_vram_inputs.pt")
    parser.add_argument("--latent_path", default="./outputs_avatar_single/low_vram_latent.pt")
    parser.add_argument("--output_dir", default="./outputs_avatar_single")
    parser.add_argument("--work_dir", default="./audio_temp_file_low_vram")
    parser.add_argument("--resolution", choices=["480p"], default="480p")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--context_parallel_size", type=int, default=4)
    parser.add_argument("--dit_subfolder", default="base_model_int8_dmd_merged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequential_block_cpu_offload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "prepare":
        prepare(parsed)
    elif parsed.mode == "denoise":
        denoise(parsed)
    else:
        decode(parsed)
