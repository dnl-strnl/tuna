import copy
import hydra
import logging as log
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from omegaconf import DictConfig
from pathlib import Path
from tuna import configure, save_model, fetch_from_hub
from tuna.lora import get_classes


@hydra.main(**configure(__file__))
def main(cfg: DictConfig):
    weights, config, tokenizer = fetch_from_hub(cfg.model)

    weights = {k:v.astype(mx.float16) for k,v in weights.items()}

    quantized_config = copy.deepcopy(config)

    model_class, model_args_class = get_classes(config=config)

    model = model_class(model_args_class.from_dict(config))
    model.load_weights(list(weights.items()), strict=False)

    q_fn = lambda m: isinstance(m, nn.Linear) and m.weight.shape[0] != 8

    nn.QuantizedLinear.quantize_module(
        model,
        cfg.group_size,
        cfg.bits,
        linear_class_predicate=q_fn,
    )

    quantization = dict(group_size=cfg.group_size, bits=cfg.bits)
    quantized_config['quantization'] = quantization
    quantized_weights = dict(tree_flatten(model.parameters()))

    model_name = f'{Path(cfg.model).stem}_q{cfg.bits}'

    save_model(model_name, weights, tokenizer, config)

    log.info(f'🔢 save {model_name=}')

    return quantized_weights, quantized_config


if __name__ == "__main__":
    main()
