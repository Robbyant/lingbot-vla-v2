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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py

from abc import ABC

import torch
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel
try:  # removed in transformers 5.x; legacy auto-load path only
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None
try:  # removed in transformers 5.x; legacy auto-load path only
    from transformers.modeling_utils import no_init_weights
except ImportError:
    import contextlib

    @contextlib.contextmanager
    def no_init_weights():
        yield
from ..utils import logging
from ..utils.import_utils import is_torch_npu_available, is_vescale_available
from .module_utils import init_empty_weights, load_model_weights
from .registry import get_registry
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Optional
logger = logging.get_logger(__name__)


class BaseModelLoader(ABC):
    def __init__(self):
        pass

    def load_model(self, model_config, **kwargs):
        raise NotImplementedError


class HuggingfaceLoader(BaseModelLoader):
    def __init__(self):
        super().__init__()

    def load_model(self, init_kwargs: dict, **kwargs):
        model_config = init_kwargs["config"]
        architecture = _get_model_arch_from_config(model_config)

        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
            load_class = AutoModelForVision2Seq
        elif "ForCausalLM" in architecture and type(model_config) in AutoModelForCausalLM._model_mapping.keys():
            load_class = AutoModelForCausalLM
        else:
            load_class = AutoModel

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)

        logger.info_rank0(
            f"Loading model from Huggingface modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if is_torch_npu_available() and init_device == "cuda":
                init_device = "npu"
            if init_device == "meta":
                with torch.device(init_device), no_init_weights():
                    logger.info_rank0("Init empty model on meta device from config without init_weights.")
                    model = load_class.from_config(**init_kwargs)
            else:
                with torch.device(init_device):
                    logger.info_rank0("Init empty model from config.")
                    model = load_class.from_config(**init_kwargs)
        else:
            if is_vescale_available() and init_device == "meta":
                from vescale.initialize.meta_init import meta_device_init

                with meta_device_init():
                    model = load_class.from_config(**init_kwargs)
            else:
                with init_empty_weights(), no_init_weights():
                    model = load_class.from_config(**init_kwargs)
            if not empty_init:
                load_model_weights(model, weights_path, init_device)

        return model


class CustomizedModelingLoader(BaseModelLoader):
    def __init__(self, model_cls: PreTrainedModel):
        super().__init__()
        self.model_cls = model_cls # model class from code_path

    def load_model(self, init_kwargs: dict, **kwargs):
        init_kwargs.pop("trust_remote_code", True)

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)
        vlm_repo_id = kwargs.pop("vlm_repo_id", None)
        post_training = kwargs.pop("post_training", False)
        adanorm_time = kwargs.pop("adanorm_time", False)
        incremental_training = kwargs.pop("incremental_training", False)

        logger.info_rank0(
            f"Loading model from customized modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if is_torch_npu_available() and init_device == "cuda":
                init_device = "npu"
            if init_device == "meta":
                with torch.device(init_device), no_init_weights():
                    logger.info_rank0("Init empty model on meta device from config without init_weights.")
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                with torch.device(init_device):
                    logger.info_rank0("Init empty model from config.")
                    model = self.model_cls._from_config(**init_kwargs)
        else:
            load_vlm_only = False
            if is_vescale_available() and init_device == "meta":
                from vescale.initialize.meta_init import meta_device_init

                with meta_device_init():
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                with init_empty_weights(), no_init_weights():
                    if 'vla' in self.model_cls.__module__:
                        model = self.model_cls(config=init_kwargs['config']).to(init_kwargs['torch_dtype'])
                        if vlm_repo_id is not None:
                            load_vlm_only = True
                    else:
                        model = self.model_cls._from_config(**init_kwargs)

            if not empty_init:
                load_model_weights(model, weights_path, init_device, load_vlm_only=load_vlm_only, post_training=post_training, incremental_training=incremental_training, adanorm_time=adanorm_time)

            # we should tie embeddings after loading weights because init_empty_weights() leads to untied weights,
            if getattr(model.config, "tie_word_embeddings", True):
                try:
                    input_embeddings = model.get_input_embeddings()
                    output_embeddings = model.get_output_embeddings()
                    output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]
                except Exception as e:
                    logger.info_rank0(f"Failed to tie embeddings: {e}")

        return model


def _get_model_arch_from_config(model_config):
    arch_name = model_config.architectures
    if isinstance(arch_name, list):
        arch_name = arch_name[0]
    return arch_name


def get_loader(model_config, force_use_huggingface):
    model_arch = _get_model_arch_from_config(model_config) # Qwen2VLForConditionalGeneration
    loader = HuggingfaceLoader()
    if not force_use_huggingface:
        model_registry = get_registry()
        if model_arch in model_registry.supported_models:
            model_cls = model_registry.get_model_cls_from_model_arch(model_arch) # <class 'veomni.models.transformers.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration'>
            loader = CustomizedModelingLoader(model_cls=model_cls)

    return loader


class VLAWeightLoader(ABC):
    """
    Base class for VLA weight loaders
    """
    def get_vlm_para_fullnames(self, model)-> set[str]:
        vlm_mod = self.get_vlm_submodule(model)
        prefix = self.get_submodule_prefix(model, vlm_mod)
        return {f"{prefix}.{n}" for n, _ in vlm_mod.named_parameters()}
    

    def get_expert_visual_para_fullnames(self, model: nn.Module)-> set[str]:
        ev_mod = self.get_expert_vision_submodule()
        if ev_mod is None:
            return set()
        prefix = self.get_submodule_prefix(model, ev_mod)
        return {f"{prefix}.{n}" for n, _ in ev_mod.named_parameters()}
    
    def map_ckpt_key(self, key, load_vlm_only: bool, post_training: bool)-> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_vlm_submodule(self, model: nn.Module)-> nn.Module:
        ...
    
    @abstractmethod
    def get_expert_vision_submodule(self):
        ...
    
    def get_submodule_prefix(self, model, submodule)->str:
        for name, mod in model.named_modules():
            if mod is submodule:
                return name
        raise ValueError(f"Submodule {type(submodule).__name__} not found in model")

class LingBotVLAWeightLoader(VLAWeightLoader):

    def get_vlm_submodule(self, model: nn.Module) -> nn.Module:
        return model.model.qwenvl_with_expert.qwenvl

    def get_expert_vision_submodule(self, model: nn.Module)-> Optional[nn.Module]:
        ev = getattr(model.model.qwenvl_with_expert, "expert_visual", None)
        return ev
    
    def map_ckpt_key(self, key: str, load_vlm_only: bool, post_training: bool)-> Optional[str]:
        if key.startswith('expert_visual.') and not post_training:
            return "model.qwenvl_with_expert." + key
        if load_vlm_only:
            return "model.qwenvl_with_expert.qwenvl." + key
        
        return key