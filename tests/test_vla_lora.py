import json
from copy import deepcopy

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2Config, Qwen2Model

from lingbotvla.utils.lora_utils import (
    configure_vla_lora,
    merge_vla_adapter_weights,
    merge_vla_lora_state_dict,
    save_vla_adapter,
)


def make_model():
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_dropout=0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2Model(config)
    model.register_buffer("e_score_correction_bias", torch.zeros(4))
    return model


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradients_freezing_and_adapter_roundtrip(tmp_path, checkpointing):
    base = make_model()
    model = deepcopy(base)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    expected = base(tokens).last_hidden_state.detach()
    config = configure_vla_lora(model, 2, 4, "q_proj,k_proj,v_proj,o_proj")
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.testing.assert_close(model(tokens).last_hidden_state, expected, rtol=0, atol=0)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in name for name in trainable)
    assert sum(p.numel() for p in trainable.values()) < sum(p.numel() for p in base.parameters())
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01)
    target = torch.randn_like(expected)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(tokens).last_hidden_state - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        optimizer.step()
    assert any(torch.count_nonzero(p.grad) for p in trainable.values())
    for name, p in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
    model.e_score_correction_bias.copy_(torch.tensor([0.1, -0.1, 0.2, -0.2]))
    model.eval()
    output = model(tokens).last_hidden_state.detach()
    assert not torch.equal(output, expected)
    save_vla_adapter(tmp_path, model.state_dict(), config)
    assert (tmp_path / "adapter_model.safetensors").stat().st_size < sum(
        p.numel() * p.element_size() for p in base.parameters()
    )
    restored = deepcopy(base)
    restored.load_state_dict(merge_vla_adapter_weights(restored.state_dict(), tmp_path), strict=True)
    merged = deepcopy(base)
    merged.load_state_dict(merge_vla_lora_state_dict(model.state_dict(), config), strict=True)
    for candidate in (restored, merged):
        candidate.eval()
        torch.testing.assert_close(candidate(tokens).last_hidden_state, output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(candidate.e_score_correction_bias, model.e_score_correction_bias)
    # Training-state restoration must preserve the next AdamW update, not just inference.
    resumed = deepcopy(base)
    configure_vla_lora(resumed, 2, 4, "q_proj,k_proj,v_proj,o_proj")
    resumed.load_state_dict(model.state_dict())
    resumed_optimizer = torch.optim.AdamW((p for p in resumed.parameters() if p.requires_grad), lr=0.01)
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    for candidate, optim in ((model, optimizer), (resumed, resumed_optimizer)):
        optim.zero_grad()
        (candidate(tokens).last_hidden_state - target).square().mean().backward()
        optim.step()
    torch.testing.assert_close(resumed(tokens).last_hidden_state, model(tokens).last_hidden_state, rtol=0, atol=0)


def test_reject_incomplete_adapter_before_modifying_model(tmp_path):
    base = make_model()
    model = deepcopy(base)
    config = configure_vla_lora(model, 2, 4, "q_proj,v_proj")
    save_vla_adapter(tmp_path, model.state_dict(), config)
    weights = load_file(tmp_path / "adapter_model.safetensors")
    prefix = config["module_names"][0]
    del weights[prefix + ".lora_A.default.weight"]
    del weights[prefix + ".lora_B.default.weight"]
    save_file(weights, tmp_path / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="missing or unexpected"):
        merge_vla_adapter_weights(base.state_dict(), tmp_path)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, make_model().state_dict()[name], rtol=0, atol=0)


def test_reject_router_and_fused_expert_targets():
    with pytest.raises(ValueError, match="VLA LoRA targets"):
        configure_vla_lora(make_model(), 2, 4, "gate_proj")


@pytest.mark.parametrize("base_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("save_dtype", [torch.float32, torch.bfloat16])
def test_adapter_preserves_training_and_export_precision(tmp_path, base_dtype, save_dtype):
    original = make_model()
    trained = deepcopy(original).to(base_dtype)
    config = configure_vla_lora(trained, 2, 4, "q_proj,v_proj")
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(std=0.01)
    save_vla_adapter(tmp_path, trained.state_dict(), config, save_dtype=save_dtype)
    merged_weights = {
        name: value.to(save_dtype) for name, value in merge_vla_lora_state_dict(trained.state_dict(), config).items()
    }
    adapter_weights = merge_vla_adapter_weights(original.state_dict(), tmp_path)
    merged, adapted = make_model(), make_model()
    merged.load_state_dict(merged_weights, strict=True)
    adapted.load_state_dict(adapter_weights, strict=True)
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(adapted(tokens).last_hidden_state, merged(tokens).last_hidden_state, rtol=0, atol=0)


@pytest.mark.parametrize("use_lora,enable_fp32", [(False, True), (True, True), (True, False)])
def test_dcp_resume_and_hf_export(tmp_path, use_lora, enable_fp32):
    import torch.distributed as dist

    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.utils.async_hf_checkpoint import AsyncHFCheckpointSaver

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        (tmp_path / "lingbotvla_cli.yaml").write_text("model: {}\ndata: {}\ntrain: {}\n")
        base = make_model()
        model = deepcopy(base)
        if use_lora:
            configure_vla_lora(model, 2, 4, "q_proj,v_proj")
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
        tokens = torch.tensor([[1, 2, 3, 4]])
        target = torch.randn(1, 4, 16)

        def step(model, optimizer):
            optimizer.zero_grad()
            (model(tokens).last_hidden_state - target).square().mean().backward()
            optimizer.step()

        step(model, optimizer)
        checkpoint = build_checkpointer("ddp", "dcp")
        state = {"model": model, "optimizer": optimizer, "extra_state": {"step": 1}}
        checkpoint.save(str(tmp_path), state, global_steps=1)
        path = tmp_path / "global_step_1"
        saver = AsyncHFCheckpointSaver(enabled=True)
        saver.submit(1, str(path), str(tmp_path), "dcp", False, enable_fp32, [model.config])
        summary = saver.wait_all_best_effort()
        assert summary["hf_success"] == 1, summary
        exported = deepcopy(base)
        exported.load_state_dict(load_file(path / "hf_ckpt/model.safetensors"), strict=True)
        if enable_fp32:
            torch.testing.assert_close(
                exported(tokens).last_hidden_state, model(tokens).last_hidden_state, rtol=1e-5, atol=1e-6
            )
        if use_lora:
            from_adapter = deepcopy(base)
            from_adapter.load_state_dict(merge_vla_adapter_weights(base.state_dict(), path / "hf_ckpt/adapter"))
            torch.testing.assert_close(
                from_adapter(tokens).last_hidden_state, exported(tokens).last_hidden_state, rtol=0, atol=0
            )
        assert "vla_lora" not in json.loads((path / "hf_ckpt/config.json").read_text())
        assert (path / "hf_ckpt/adapter/adapter_model.safetensors").exists() == use_lora
        restored = deepcopy(base)
        if use_lora:
            configure_vla_lora(restored, 2, 4, "q_proj,v_proj")
        restored_optimizer = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=0.01)
        state = {"model": restored, "optimizer": restored_optimizer, "extra_state": {}}
        checkpoint.load(str(path), state)
        assert state["extra_state"]["step"] == 1
        step(model, optimizer)
        step(restored, restored_optimizer)
        torch.testing.assert_close(restored(tokens).last_hidden_state, model(tokens).last_hidden_state, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
