import glob
import json
from huggingface_hub import HfApi, ModelCard, logging, snapshot_download
import mlx.core as mx
import os
from pathlib import Path
import transformers


def configure(str_path, config_dir='config', max_levels=6):
    path = Path(str_path).resolve()
    print(path)
    proj_root = None

    for i in range(1, max_levels):
        potential_root = path.parents[i-1]
        if (potential_root / config_dir).exists():
            proj_root = potential_root
            break

    if proj_root is None:
        raise FileNotFoundError()

    conf_path = proj_root / config_dir
    conf_dict = dict(config_path=str(conf_path), config_name=path.stem)
    return conf_dict


def fetch_from_hub(hf_path: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        allow_patterns=['*.json', '*.safetensors', 'tokenizer.model'],
    )
    weight_files = glob.glob(f'{model_path}/*.safetensors')
    if len(weight_files) == 0:
        raise FileNotFoundError(f'no safetensors found in {model_path}')

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def upload_to_hub(path: str, name: str, hf_path: str):

    repo_id = f'mlx-community/{name}'

    card = ModelCard.load(hf_path)
    card.data.tags = ['mlx'] if card.data.tags is None else card.data.tags + ['mlx']
    card.text = f'''
    # {name}
    This model was converted to MLX format from [`{hf_path}`]().
    Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
    ## Use with mlx
    ```bash
    pip install mlx
    git clone https://github.com/ml-explore/mlx-examples.git
    cd mlx-examples/llms/hf_llm
    python generate.py --model {repo_id} --prompt 'My name is'
    ```
    '''
    card.save(os.path.join(path, 'README.md'))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type='model',
    )


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        'model-{:05d}-of-{:05d}.safetensors'
        if shards_count > 1
        else 'model.safetensors'
    )

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(str(save_dir / shard_name), shard)

    tokenizer.save_pretrained(save_dir)

    with open(save_dir / 'config.json', 'w') as fid:
        json.dump(config, fid, indent=4)
