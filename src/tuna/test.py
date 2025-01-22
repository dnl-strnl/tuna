import mlx.core as mx
import mlx.nn as nn
from typing import Generator


def sample(prompt: mx.array, model: nn.Module, temperature: float = 0.0) -> Generator[mx.array, None, None]:

    def sample_fn(logits: mx.array) -> mx.array:
        return mx.argmax(logits, axis=-1) if temperature == 0 else \
            mx.random.categorical(logits * (1 / temperature))

    output, cache = prompt, None
    while True:
        logits, cache = model(output[None], cache=cache)
        logits = logits[:, -1, :]
        output = sample_fn(logits)
        yield output


def generate(model, tokenizer, prompt, temperature=1, max_tokens=2**11):
    print(prompt, end='', flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(sample(prompt, model, temperature), range(max_tokens)):

        if token == tokenizer.eos_token_id: break

        tokens.append(token.item())

        output = tokenizer.decode(tokens)

        if len(output) - skip > 1:
            print(output[skip:-1], end='', flush=True)
            skip = len(output) - 1

    print(tokenizer.decode(tokens)[skip:], flush=True)
