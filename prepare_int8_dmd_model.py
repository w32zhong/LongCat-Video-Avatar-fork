import argparse
from pathlib import Path

from longcat_video.modules.quantization import (
    load_quantized_dit,
    merge_lora_into_quantized_model,
    save_quantized_state_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge the Avatar 1.5 DMD LoRA into the INT8 DiT checkpoint."
    )
    parser.add_argument("--checkpoint_dir", required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    base_dir = checkpoint_dir / "base_model_int8"
    lora_path = checkpoint_dir / "lora" / "dmd_lora.safetensors"
    output_dir = checkpoint_dir / "base_model_int8_dmd_merged"

    required_files = [
        base_dir / "config.json",
        base_dir / "quantized_model.safetensors.index.json",
        lora_path,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))

    output_index = output_dir / "quantized_model.safetensors.index.json"
    if output_index.is_file():
        print(f"Already prepared: {output_dir}")
        return

    print(f"Loading INT8 model: {base_dir}", flush=True)
    model = load_quantized_dit(
        str(checkpoint_dir),
        subfolder="base_model_int8",
        low_cpu_mem_usage=True,
        device="cpu",
    )
    print(f"Merging DMD LoRA: {lora_path}", flush=True)
    merged_count = merge_lora_into_quantized_model(model, str(lora_path))
    print(f"Saving merged model: {output_dir}", flush=True)
    save_quantized_state_dict(
        model,
        str(output_dir),
        config_source_dir=str(base_dir),
    )
    print(f"Prepared {output_dir} ({merged_count} LoRA modules merged)")


if __name__ == "__main__":
    main()
