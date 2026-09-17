"""Select and operate on the best available PyTorch device backend."""

import importlib.util
import os

import torch


def _force_cpu() -> bool:
    return (
        os.environ.get("DEVICE_SELECT_FORCE_CPU") == "1"
        or os.environ.get("XPU_COMPAT_FORCE_CPU") == "1"
    )


def _accelerator_module():
    if _force_cpu():
        return None
    for name in ("xpu", "cuda"):
        accelerator = getattr(torch, name, None)
        if accelerator is not None and accelerator.is_available():
            return accelerator
    return None


def has_accelerator() -> bool:
    return _accelerator_module() is not None


def device_type() -> str:
    accelerator = _accelerator_module()
    if accelerator is None:
        return "cpu"
    return accelerator.__name__.rsplit(".", 1)[-1]


def set_device(device) -> None:
    accelerator = _accelerator_module()
    if accelerator is not None:
        accelerator.set_device(device)


def synchronize() -> None:
    accelerator = _accelerator_module()
    if accelerator is not None:
        accelerator.synchronize()


def event(enable_timing: bool = False):
    accelerator = _accelerator_module()
    if accelerator is not None:
        return accelerator.Event(enable_timing=enable_timing)

    class _NoOpEvent:
        def record(self, *args, **kwargs):
            return None

        def synchronize(self):
            return None

        def elapsed_time(self, *args, **kwargs):
            return 0.0

    return _NoOpEvent()


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    accelerator = _accelerator_module()
    if accelerator is not None:
        accelerator.manual_seed_all(seed)


def attn_implementation(name: str) -> str:
    """Use SDPA when the requested optional attention backend is unavailable."""
    if name == "flash_attention_2" and _find_spec("flash_attn") is None:
        return "sdpa"
    return name


def _find_spec(name: str):
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError):
        return None