from dataclasses import dataclass
import glob
import inspect
import json
from huggingface_hub import snapshot_download
import logging as log
import math
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
from pathlib import Path
import transformers
from typing import Dict, Generator, List, Optional, Tuple, Union

from tuna.lora.models import ModelArgs
from tuna.lora.models import llama
from tuna.lora.models import mixtral
from tuna.lora.models import phi2
model_mapping = {
    'llama': llama,
    # mistral is compatible with llama
    'mistral': llama,
    'mixtral': mixtral,
    'phi': phi2,
}

def get_classes(config: dict):
    model_type = config['model_type']
    if model_type not in model_mapping:
        msg = f'Model type {model_type} not supported.'
        logging.error(msg)
        raise ValueError(msg)
    arch = model_mapping[model_type]
    return arch.Model, arch.ModelArgs

class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear: nn.Linear, rank: int = 8):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        return lora_lin

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = 'bias' in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        lora_b = (self.scale * self.lora_b.T).astype(dtype)
        lora_a = self.lora_a.T.astype(dtype)
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized and not de_quantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int = 8,
        bias: bool = False,
        scale: float = 20.0,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dims, output_dims, bias=bias)

        self.scale = scale

        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, lora_rank),
        )
        self.lora_b = mx.zeros(shape=(lora_rank, output_dims))

    def __call__(self, x):
        dtype = self.linear.weight.dtype
        if isinstance(self.linear, nn.QuantizedLinear):
            dtype = self.linear.scales.dtype
        y = self.linear(x.astype(dtype))
        z = (x @ self.lora_a) @ self.lora_b
        return y + self.scale * z


def freeze_layers(model, layers:int):
    model.freeze()
    for l in model.model.layers[len(model.model.layers) - layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, 'block_sparse_moe'):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    total_params = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    return train_params, total_params


def load_lora(path_or_hf_repo: str):
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=['*.json', '*.safetensors', 'tokenizer.model'],
            )
        )

    with open(model_path / 'config.json', 'r') as f:
        config = json.loads(f.read())
        quantization = config.get('quantization', None)

    weight_files = glob.glob(str(model_path / '*.safetensors'))
    if len(weight_files) == 0:
        raise FileNotFoundError('no safetensors found in {}'.format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model_class, model_args_class = get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
            and m.weight.shape[0] != 8,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config
