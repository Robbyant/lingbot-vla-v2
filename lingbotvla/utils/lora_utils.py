# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def freeze_parameters(model: nn.Module):
    # Freeze parameters
    model.requires_grad_(False)
    model.eval()
    model.train()


def add_lora_to_model(
    model: nn.Module,
    lora_rank=4,
    lora_alpha=4,
    lora_target_modules="q,k,v,o,ffn.0,ffn.2",
    init_lora_weights="kaiming",
    pretrained_lora_path=None,
    state_dict_converter=None,
    lora_target_modules_support=None,
):
    model.lora_alpha = lora_alpha
    if init_lora_weights == "kaiming":
        init_lora_weights = True

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )

    for lora_target_module in lora_config.target_modules:
        if lora_target_module not in lora_target_modules_support:
            raise ValueError(f"lora_target_module {lora_target_module} not in lora_target_modules_support")

    model = inject_adapter_in_model(lora_config, model)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(dtype=torch.float32)

    # Lora pretrained lora weights
    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(
            f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
        )


def configure_vla_lora(model, rank, alpha, target_modules):
    """Adapt attention projections in the language backbone and action expert."""
    supported = {"q_proj", "k_proj", "v_proj", "o_proj"}
    targets = target_modules.split(",")
    if not targets or not set(targets) <= supported:
        raise ValueError(f"VLA LoRA targets must be selected from {sorted(supported)}.")
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive.")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("VLA LoRA requires materialized base weights; use init_device=cuda.")
    add_lora_to_model(
        model,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target_modules=target_modules,
        lora_target_modules_support=supported,
    )
    # Loss-free routing updates these buffers even when base parameters are frozen.
    buffer_names = [name for name in model.state_dict() if name.endswith("e_score_correction_bias")]
    model.config.vla_lora = {
        "rank": rank,
        "alpha": alpha,
        "target_modules": target_modules,
        "module_names": [name for name, module in model.named_modules() if hasattr(module, "lora_A")],
        "buffer_names": buffer_names,
    }
    return model.config.vla_lora


def save_vla_adapter(output_dir, state_dict, config, save_dtype=torch.float32):
    """Save adapter tensors and the routing state needed to reproduce inference."""
    os.makedirs(output_dir, exist_ok=True)
    weights = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in state_dict.items()
        if ".lora_A.default.weight" in name or ".lora_B.default.weight" in name or name in config["buffer_names"]
    }
    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    # The training loader materializes all base parameters in one dtype.
    base_weight = state_dict[config["module_names"][0] + ".base_layer.weight"]
    config = {
        **config,
        "base_dtype": str(base_weight.dtype).split(".")[-1],
        "save_dtype": str(save_dtype).split(".")[-1],
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def merge_vla_lora_state_dict(state_dict, config):
    """Convert the injected model's state into the ordinary inference model's keys."""
    merged = {
        name.replace(".base_layer.", "."): value
        for name, value in state_dict.items()
        if ".lora_A." not in name and ".lora_B." not in name
    }
    for name, a in state_dict.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        prefix = name.removesuffix(".lora_A.default.weight")
        b = state_dict[prefix + ".lora_B.default.weight"]
        weight_name = prefix + ".weight"
        weight = merged[weight_name]
        merged[weight_name] = (weight.float() + (b.float() @ a.float()) * (config["alpha"] / config["rank"])).to(
            weight.dtype
        )
    return merged


def merge_vla_adapter_weights(state, adapter_path):
    """Merge into the original checkpoint before deployment casts its weights."""
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        config = json.load(f)
    adapter = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    base_dtype = getattr(torch, config["base_dtype"])
    save_dtype = getattr(torch, config["save_dtype"])
    expected = set(config["buffer_names"])
    for prefix in config["module_names"]:
        expected.update((prefix + ".lora_A.default.weight", prefix + ".lora_B.default.weight"))
    if not config["module_names"] or expected != set(adapter):
        raise ValueError("Adapter has missing or unexpected tensors.")
    updates = {}
    for name, a in adapter.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        prefix = name.removesuffix(".lora_A.default.weight")
        b_name = prefix + ".lora_B.default.weight"
        b = adapter[b_name]
        weight_name = prefix + ".weight"
        weight = state[weight_name].to(base_dtype)
        if a.shape[0] != config["rank"] or b.shape[1] != config["rank"] or (b.shape[0], a.shape[1]) != weight.shape:
            raise ValueError(f"Adapter shape does not match base weight {weight_name}.")
        updates[weight_name] = (
            weight.float()
            + (b.to(weight.device).float() @ a.to(weight.device).float()) * (config["alpha"] / config["rank"])
        ).to(weight.dtype)
    for name in config["buffer_names"]:
        if name not in state or adapter[name].shape != state[name].shape:
            raise ValueError(f"Adapter routing buffer does not match base model: {name}.")
        updates[name] = adapter[name]
    merged = {name: value.to(base_dtype) for name, value in state.items()}
    merged.update(updates)
    return {name: value.to(save_dtype) for name, value in merged.items()}


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict
