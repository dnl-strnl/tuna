import datetime
import hydra
import logging as log
import math
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import time
from tuna import configure
from tuna.data import batch_iterator, Dataset
from tuna.lora import load_lora, freeze_layers
from tuna.test import generate


def loss(model, inputs, targets, lengths):

    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()

    ce = ce.sum() / ntoks
    return ce, ntoks


def evaluate(model, dataset, loss, tokenizer, batch_size, batches):
    all_losses, num_tokens = [], 0

    iterator = batch_iterator(dataset, tokenizer, batch_size)

    for itr, batch in zip(range(batches), iterator):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        num_tokens += toks.item()

    return np.sum(all_losses) / num_tokens


def train_log(itr, ntok, train_loss, steps_per, start, stop):
    itrs_per_sec = steps_per / (stop - start)
    toks_per_sec = float(ntok) / (stop - start)
    return f'{itr=:7}: {train_loss=:.3f}, {itrs_per_sec=:.3f}, {toks_per_sec=:.3f}'


@hydra.main(**configure(__file__))
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)

    model, dataset, adapter = cfg.model, cfg.dataset, cfg.adapter
    batch_size, batches = cfg.train.batch_size, cfg.test.batches
    report = cfg.train.steps_per_report

    log.info(f'load 🧠 {model=}')

    if cfg.mode == 'lora': model, tknzr, _ = load_lora(model)
    else: raise NotImplementedError()

    # freeze all layers other than LoRA linear layers
    train_params, total_params = freeze_layers(model, cfg.lora.layers)
    log.info(f'freeze layers ❄️  {train_params=:.3f}M / {total_params=:.3f}M')

    train_set = Dataset(Path(dataset) / 'train')
    test_set = Dataset(Path(dataset) / 'test')
    val_set = Dataset(Path(dataset) / 'valid')
    log.info(f'load 🧮 {dataset=}')

    # resume adapter training from checkpoint
    if Path(str(adapter)).exists():
        model.load_weights(adapter, strict=False)
        log.info(f'load 🔌 {adapter=}')

    eval_args = lambda dataset:dict(model=model, loss=loss, tokenizer=tknzr,
        dataset=dataset, batch_size=batch_size, batches=batches
    )
    trainable_parameters = lambda:dict(tree_flatten(model.trainable_parameters()))

    if cfg.train:
        tune_init = str(datetime.datetime.now())
        log.info(f'🐟 {tune_init=}')

        optim = hydra.utils.instantiate(cfg.optimizer)
        loss_and_grad = nn.value_and_grad(model, loss)

        iterator = batch_iterator(train_set, tknzr, batch_size, train=True)
        losses, ntok = [], 0
        start = time.perf_counter()

        for itr, batch in zip(range(1, cfg.train.itrs + 1), iterator):

            # forward and backward pass
            (loss_value, toks), grad = loss_and_grad(model, *batch)
            # model update
            optim.update(model, grad)
            mx.eval(model.parameters(), optim.state, loss_value)
            # record loss
            losses.append(loss_value.item())
            ntok += toks.item()

            # report train loss
            if itr % report == 0:
                train_loss = np.mean(losses)
                stop = time.perf_counter()
                log.info(train_log(itr, ntok, train_loss, report, start, stop))
                losses, ntok = [], 0
                start = time.perf_counter()

            # report validation loss
            if itr == 1 or itr % cfg.train.steps_per_eval == 0:
                stop = time.perf_counter()

                val_loss = float(evaluate(**eval_args(val_set)))
                duration = float(time.perf_counter() - stop)

                log.info(f'{itr=:7}: {val_loss=:.3f}, {duration=:.3f}')
                start = time.perf_counter()

            # save adapter weights
            if itr % cfg.train.save_every == 0:
                mx.savez(adapter, **trainable_parameters())
                log.info(f'{itr=:7}: save {adapter=}')

        tune_stop = str(datetime.datetime.now())
        log.info(f'🎣 {tune_stop=}')

        # save adapter weights
        log.info(f'save 🔌 {adapter=}')
        mx.savez(adapter, **trainable_parameters())
    else:
        # load adapter weights
        log.info(f'load 🔌 {adapter=}')
        model.load_weights(adapter, strict=False)

    if cfg.test.batches > 0:
        log.info('test 🧪')
        model.eval()
        test_loss = evaluate(**eval_args(test_set))
        test_perplexity = math.exp(test_loss)
        log.info(f'{test_loss=:.3f}, {test_perplexity=:.3f}')

    if cfg.prompt is not None:
        log.info('inference 🎏')
        generate(model, tknzr, cfg.prompt, **dict(cfg.test.sample))


if __name__ == '__main__':
    main()
