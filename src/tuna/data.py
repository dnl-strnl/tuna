import json
import mlx.core as mx
import numpy as np
from pathlib import Path


class Dataset:
    def __init__(self, path: Path, key: str = 'text'):
        split = Path(f'{path}.jsonl')

        if not split.exists():
            self._data = None
        else:
            with open(split, 'r') as f:
                self._data = [json.loads(l) for l in f]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def batch_iterator(dataset, tokenizer, batch_size=2, max_tok=2**11, train=False):
    while True:
        indices = np.arange(len(dataset))

        if train:
            indices = np.random.permutation(indices)

        for i in range(0, len(indices) - batch_size + 1, batch_size):
            batch = [tokenizer.encode(dataset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # pad to the max sequence length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]

            batch = mx.array(batch_arr)

            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train: break
