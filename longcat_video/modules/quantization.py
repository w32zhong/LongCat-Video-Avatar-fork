import os
import json
import shutil
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file, load_file


class QuantizedLinear(nn.Module):
    """INT8 weight-only quantized linear layer with per-channel symmetric quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight to input dtype for computation
        compute_dtype = x.dtype
        weight = self.weight_int8.to(compute_dtype) * self.weight_scale.to(compute_dtype).unsqueeze(1)
        bias = self.bias.to(compute_dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """Quantize a standard Linear layer to INT8."""
        has_bias = linear.bias is not None
        ql = cls(linear.in_features, linear.out_features, bias=has_bias)

        weight = linear.weight.data.float()
        # Per-channel symmetric quantization
        scale = weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weight_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        ql.weight_int8 = weight_int8
        ql.weight_scale = scale
        if has_bias:
            ql.bias = linear.bias.data.to(torch.bfloat16)
        return ql

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# Layers to skip quantization (sensitive to precision)
DEFAULT_SKIP_PATTERNS = {
    "final_layer.linear",  # Final output projection, precision-sensitive
}


def quantize_model(model: nn.Module, skip_patterns: Optional[Set[str]] = None) -> nn.Module:
    """Replace all nn.Linear layers in the model with QuantizedLinear (INT8 weight-only).

    Args:
        model: The model to quantize (modified in-place).
        skip_patterns: Set of module name patterns to skip. If None, uses DEFAULT_SKIP_PATTERNS.

    Returns:
        The quantized model.
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module should be skipped
            should_skip = any(pattern in name for pattern in skip_patterns)
            if not should_skip:
                modules_to_replace[name] = module

    for name, linear in modules_to_replace.items():
        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        # Replace with quantized version
        setattr(parent, parts[-1], QuantizedLinear.from_linear(linear))

    return model


def save_quantized_state_dict(model: nn.Module, save_dir: str, config_source_dir: Optional[str] = None):
    """Save the quantized model's state dict using safetensors format.

    Saves quantized weights in shards to avoid single large files.

    Args:
        model: The quantized model.
        save_dir: Directory to save quantized weights.
        config_source_dir: If provided, copy config.json from this directory.
    """
    os.makedirs(save_dir, exist_ok=True)

    state_dict = model.state_dict()

    # Split into shards (~4GB each for manageable files)
    max_shard_size = 4 * 1024 * 1024 * 1024  # 4GB

    shards = []
    current_shard = {}
    current_size = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    # Save each shard
    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in state_dict.values())}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = f"quantized_model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(save_dir, shard_name))
        for key in shard:
            index["weight_map"][key] = shard_name

    # Save index
    index_path = os.path.join(save_dir, "quantized_model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    # Copy config if source provided
    if config_source_dir:
        src_config = os.path.join(config_source_dir, "config.json")
        if os.path.exists(src_config):
            shutil.copy2(src_config, os.path.join(save_dir, "config.json"))

    # Save quantization metadata
    quant_config = {
        "quantization_method": "int8_per_channel_symmetric",
        "skip_patterns": list(DEFAULT_SKIP_PATTERNS),
        "description": "Weight-only INT8 quantization with per-channel symmetric scaling"
    }
    with open(os.path.join(save_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    print(f"Saved {len(shards)} shard(s) to {save_dir}")


def _replace_linear_with_empty_quantized(model: nn.Module):
    """Replace Linear layers without first materializing their quantized weights."""
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = any(pattern in name for pattern in DEFAULT_SKIP_PATTERNS)
            if not should_skip:
                modules_to_replace[name] = QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )

    for name, quantized_linear in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], quantized_linear)


def merge_lora_into_quantized_model(model: nn.Module, lora_path: str, multiplier: float = 1.0) -> int:
    """Merge a LoRA checkpoint into INT8 weights and requantize them in-place.

    This removes the need to keep the LoRA modules resident during inference.
    It is intended for weight-only INT8 inference where the base weights are
    already approximate.
    """
    merged = 0
    with safe_open(lora_path, framework="pt", device="cpu") as lora_file:
        keys = set(lora_file.keys())
        down_keys = sorted(key for key in keys if key.endswith(".lora_down.weight"))

        for down_key in down_keys:
            prefix = down_key[: -len(".lora_down.weight")]
            module_name = prefix.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")

            module = model
            for part in module_name.split("."):
                module = getattr(module, part)
            if not isinstance(module, QuantizedLinear):
                raise TypeError(f"LoRA target {module_name} is not QuantizedLinear: {type(module).__name__}")

            down = lora_file.get_tensor(down_key).float()
            block_prefix = prefix + ".lora_up.blocks."
            block_keys = sorted(
                (key for key in keys if key.startswith(block_prefix) and key.endswith(".weight")),
                key=lambda key: int(key[len(block_prefix):].split(".", 1)[0]),
            )
            if block_keys:
                rank = down.shape[0] // len(block_keys)
                delta = torch.cat(
                    [
                        lora_file.get_tensor(key).float()
                        @ down[index * rank : (index + 1) * rank]
                        for index, key in enumerate(block_keys)
                    ],
                    dim=0,
                )
            else:
                up = lora_file.get_tensor(prefix + ".lora_up.weight").float()
                delta = up @ down

            alpha_key = prefix + ".alpha_scale"
            alpha_scale = float(lora_file.get_tensor(alpha_key)) if alpha_key in keys else 1.0
            base_weight = module.weight_int8.float() * module.weight_scale.float().unsqueeze(1)
            merged_weight = base_weight.add_(delta, alpha=multiplier * alpha_scale)
            scale = merged_weight.abs().amax(dim=1).clamp(min=1e-8) / 127.0
            module.weight_int8 = (merged_weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
            module.weight_scale = scale
            merged += 1

    return merged


def load_quantized_dit(
    checkpoint_dir: str,
    subfolder: str = "base_model_int8",
    low_cpu_mem_usage: bool = False,
    device: Optional[str] = None,
    **kwargs,
):
    """Load a quantized DiT model.

    Args:
        checkpoint_dir: Base checkpoint directory.
        subfolder: Subfolder containing quantized weights (default: 'base_model_int8').
        **kwargs: Additional kwargs passed to the model constructor (e.g., cp_split_hw).

    Returns:
        The quantized DiT model ready for inference.
    """
    from .avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    quantized_dir = os.path.join(checkpoint_dir, subfolder)

    # Load config
    config_path = os.path.join(quantized_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Remove non-constructor keys
    config.pop("_class_name", None)
    config.pop("architectures", None)
    config.pop("_diffusers_version", None)
    config.pop("model_max_length", None)

    # Override with kwargs
    config.update(kwargs)

    if low_cpu_mem_usage:
        with torch.device("meta"):
            model = LongCatVideoAvatarTransformer3DModel(**config)
            _replace_linear_with_empty_quantized(model)
    else:
        model = LongCatVideoAvatarTransformer3DModel(**config)
        _replace_linear_with_empty_quantized(model)

    # Load quantized state dict
    index_path = os.path.join(quantized_dir, "quantized_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        # Load from shards
        shard_files = set(index["weight_map"].values())
        if low_cpu_mem_usage:
            expected_keys = set(model.state_dict().keys())
            loaded_keys = set()
            target_device = device or "cpu"
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(quantized_dir, shard_file)
                shard_dict = load_file(shard_path, device=target_device)
                model.load_state_dict(shard_dict, strict=False, assign=True)
                loaded_keys.update(shard_dict.keys())
                del shard_dict
            missing_keys = expected_keys - loaded_keys
            unexpected_keys = loaded_keys - expected_keys
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    f"Quantized checkpoint key mismatch: missing={sorted(missing_keys)[:10]}, "
                    f"unexpected={sorted(unexpected_keys)[:10]}"
                )
            state_dict = None
        else:
            state_dict = {}
            for shard_file in sorted(shard_files):
                shard_path = os.path.join(quantized_dir, shard_file)
                shard_dict = load_file(shard_path, device="cpu")
                state_dict.update(shard_dict)
    else:
        # Single file fallback
        files = [f for f in os.listdir(quantized_dir) if f.endswith(".safetensors") and "index" not in f]
        state_dict = {}
        for f in sorted(files):
            shard_dict = load_file(os.path.join(quantized_dir, f), device="cpu")
            state_dict.update(shard_dict)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Cast non-quantized parameters (Conv3d, LayerNorm, etc.) to bfloat16
    # QuantizedLinear buffers (int8, float32 scale) are kept as-is
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)

    return model
