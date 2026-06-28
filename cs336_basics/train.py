import os
import random
import time
import dataclasses
from collections.abc import Mapping

import torch
import numpy as np
import wandb

from cs336_basics.config import parse_train_config
from cs336_basics.model import AdamW, BasicsTransformerLM
from cs336_basics.model import load_checkpoint, save_checkpoint
from cs336_basics.model import (
    gradient_clipping,
    cosin_learn_rate_schedule,
    get_batch,
    cross_entropy,
)


def load_datasets(input_path: str):
    return np.load(input_path, mmap_mode="r")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_optimizer_lr(optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cfg_to_dict(cfg):
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)

    if isinstance(cfg, Mapping):
        return {k: cfg_to_dict(v) for k, v in cfg.items()}

    if isinstance(cfg, (list, tuple)):
        return [cfg_to_dict(v) for v in cfg]

    if hasattr(cfg, "__dict__"):
        return {
            k: cfg_to_dict(v)
            for k, v in vars(cfg).items()
            if not k.startswith("_")
        }

    return cfg


def get_cfg_value(obj, name: str, default=None):
    return getattr(obj, name, default)


def init_wandb(cfg):
    use_wandb = get_cfg_value(cfg.logging, "use_wandb", False)
    if not use_wandb:
        return None

    run = wandb.init(
        project=get_cfg_value(cfg.logging, "wandb_project", "cs336-basics"),
        name=get_cfg_value(cfg.logging, "wandb_run_name", None),
        entity=get_cfg_value(cfg.logging, "wandb_entity", None),
        mode=get_cfg_value(cfg.logging, "wandb_mode", None),
        config=cfg_to_dict(cfg),
    )

    # 自定义横轴：step
    wandb.define_metric("step")
    wandb.define_metric("train_step/*", step_metric="step")
    wandb.define_metric("val_step/*", step_metric="step")
    wandb.define_metric("lr", step_metric="step")
    wandb.define_metric("tokens", step_metric="step")

    # 自定义横轴：wall-clock time
    wandb.define_metric("wall_clock_seconds")
    wandb.define_metric("wall_clock_minutes", step_metric="wall_clock_seconds")
    wandb.define_metric("train_time/*", step_metric="wall_clock_seconds")
    wandb.define_metric("val_time/*", step_metric="wall_clock_seconds")

    # 可选：以 tokens 为横轴，方便比较不同 batch_size/context_length 的实验
    wandb.define_metric("train_tokens/*", step_metric="tokens")
    wandb.define_metric("val_tokens/*", step_metric="tokens")

    return run


@torch.no_grad()
def evaluate_loss(
    model,
    val_data,
    batch_size: int,
    context_length: int,
    device,
    eval_batches: int,
):
    """
    周期性评估 validation loss。

    注意：
    - eval 阶段要 model.eval()
    - 不需要 backward
    - eval 完要切回 model.train()
    """
    model.eval()

    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(
            val_data,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

@torch.no_grad()
def get_grad_stats(parameters):
    total_sq = None
    max_abs = None
    all_finite = None

    for p in parameters:
        if p.grad is None:
            continue

        g = p.grad.detach().float()

        cur_sq = torch.sum(g * g)
        cur_max_abs = torch.max(torch.abs(g))
        cur_finite = torch.isfinite(g).all()

        total_sq = cur_sq if total_sq is None else total_sq + cur_sq
        max_abs = cur_max_abs if max_abs is None else torch.maximum(max_abs, cur_max_abs)
        all_finite = cur_finite if all_finite is None else (all_finite & cur_finite)

    if total_sq is None:
        return {
            "grad_norm": 0.0,
            "grad_max_abs": 0.0,
            "grad_is_finite": True,
        }

    return {
        "grad_norm": float(torch.sqrt(total_sq).item()),
        "grad_max_abs": float(max_abs.item()),
        "grad_is_finite": bool(all_finite.item()),
    }

def main():
    cfg = parse_train_config()

    set_seed(cfg.runtime.seed)

    device = torch.device(cfg.runtime.device)

    train_data = load_datasets(cfg.data.train_path)

    val_data = None
    if cfg.data.val_path is not None:
        val_data = load_datasets(cfg.data.val_path)

    model = BasicsTransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    )
    model = model.to(device)

    optimizer = AdamW(
        params=model.parameters(),
        lr=cfg.optimizer.lr,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    wandb_run = init_wandb(cfg)

    start_iter = 0
    if cfg.ckpt.resume and os.path.exists(cfg.ckpt.checkpoint_path):
        start_iter = load_checkpoint(
            cfg.ckpt.checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

    model.train()

    start_time = time.perf_counter()

    train_loss_ema = None
    train_loss_ema_beta = 0.95
    tokens_per_step = cfg.train.batch_size * cfg.model.context_length

    for step in range(start_iter + 1, cfg.train.max_iter + 1):
        should_log = step % cfg.logging.log_every == 0

        lr = cosin_learn_rate_schedule(
            it=step,
            max_learning_rate=cfg.lr.max_learning_rate,
            min_learning_rate=cfg.lr.min_learning_rate,
            warmup_iters=cfg.lr.warmup_iters,
            cosine_cycle_iters=cfg.lr.cosine_cycle_iters,
        )
        set_optimizer_lr(optimizer, lr)

        x, y = get_batch(
            train_data,
            batch_size=cfg.train.batch_size,
            context_length=cfg.model.context_length,
            device=device,
        )

        logits = model(x)
        loss = cross_entropy(logits, y)

        loss_is_finite = bool(torch.isfinite(loss.detach()).item())
        if not loss_is_finite:
            print(f"step {step} | non-finite loss: {loss.item()}")

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "train_step/loss": float(loss.detach().item()),
                        "train_step/loss_is_finite": 0,
                        "train_step/diverged": 1,
                    },
                    step=step,
                )

            raise FloatingPointError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        optimizer.zero_grad()
        loss.backward()

        grad_stats_before_clip = None
        grad_stats_after_clip = None

        if should_log:
            grad_stats_before_clip = get_grad_stats(model.parameters())

            if not grad_stats_before_clip["grad_is_finite"]:
                print(f"step {step} | non-finite gradient before clipping")

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "step": step,
                            "train_step/loss": float(loss.detach().item()),
                            "train_step/loss_is_finite": 1,
                            "train_step/grad_is_finite": 0,
                            "train_step/diverged": 1,
                            "train_step/grad_norm_before_clip": grad_stats_before_clip["grad_norm"],
                            "train_step/grad_max_abs_before_clip": grad_stats_before_clip["grad_max_abs"],
                        },
                        step=step,
                    )

                raise FloatingPointError(
                    f"Non-finite gradient at step {step}"
                )

        gradient_clipping(
            model.parameters(),
            max_l2_norm=cfg.train.grad_clip,
        )

        if should_log:
            grad_stats_after_clip = get_grad_stats(model.parameters())

        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_clock_seconds = time.perf_counter() - start_time
        wall_clock_minutes = wall_clock_seconds / 60.0

        tokens = step * tokens_per_step
        processed_tokens_this_run = (step - start_iter) * tokens_per_step
        tokens_per_second = processed_tokens_this_run / max(wall_clock_seconds, 1e-8)
        steps_per_second = (step - start_iter) / max(wall_clock_seconds, 1e-8)

        if should_log:
            train_loss = loss.item()
            train_ppl = float(torch.exp(loss.detach()).item())

            if train_loss_ema is None:
                train_loss_ema = train_loss
            else:
                train_loss_ema = (
                    train_loss_ema_beta * train_loss_ema
                    + (1.0 - train_loss_ema_beta) * train_loss
                )

            print(
                f"step {step} | "
                f"train loss {train_loss:.4f} | "
                f"loss ema {train_loss_ema:.4f} | "
                f"train ppl {train_ppl:.4f} | "
                f"grad {grad_stats_before_clip['grad_norm']:.4f} "
                f"-> {grad_stats_after_clip['grad_norm']:.4f} | "
                f"grad max {grad_stats_before_clip['grad_max_abs']:.3e} | "
                f"lr {lr:.6e} | "
                f"tok/s {tokens_per_second:.1f} | "
                f"time {wall_clock_minutes:.2f} min"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "wall_clock_seconds": wall_clock_seconds,
                        "wall_clock_minutes": wall_clock_minutes,
                        "tokens": tokens,
                        "lr": lr,

                        # 以 step 为横轴
                        "train_step/loss": train_loss,
                        "train_step/loss_ema": train_loss_ema,
                        "train_step/perplexity": train_ppl,

                        # finite / divergence 诊断
                        "train_step/loss_is_finite": int(loss_is_finite),
                        "train_step/grad_is_finite": int(
                            grad_stats_before_clip["grad_is_finite"]
                        ),
                        "train_step/diverged": 0,

                        # 梯度诊断
                        "train_step/grad_norm_before_clip": grad_stats_before_clip["grad_norm"],
                        "train_step/grad_norm_after_clip": grad_stats_after_clip["grad_norm"],
                        "train_step/grad_max_abs_before_clip": grad_stats_before_clip["grad_max_abs"],
                        "train_step/grad_max_abs_after_clip": grad_stats_after_clip["grad_max_abs"],

                        # 速度诊断
                        "train_step/tokens_per_second": tokens_per_second,
                        "train_step/steps_per_second": steps_per_second,

                        # 以 wall-clock time 为横轴
                        "train_time/loss": train_loss,
                        "train_time/loss_ema": train_loss_ema,
                        "train_time/perplexity": train_ppl,
                        "train_time/grad_norm_before_clip": grad_stats_before_clip["grad_norm"],
                        "train_time/grad_norm_after_clip": grad_stats_after_clip["grad_norm"],
                        "train_time/tokens_per_second": tokens_per_second,
                        "train_time/steps_per_second": steps_per_second,
                    },
                    step=step,
                )

        if (
            val_data is not None
            and step % cfg.logging.eval_every == 0
        ):
            val_loss = evaluate_loss(
                model=model,
                val_data=val_data,
                batch_size=cfg.train.batch_size,
                context_length=cfg.model.context_length,
                device=device,
                eval_batches=cfg.logging.eval_batches,
            )
            val_ppl = float(np.exp(val_loss))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            wall_clock_seconds = time.perf_counter() - start_time
            wall_clock_minutes = wall_clock_seconds / 60.0

            print(
                f"step {step} | "
                f"val loss {val_loss:.4f} | "
                f"val ppl {val_ppl:.4f} | "
                f"time {wall_clock_minutes:.2f} min"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "wall_clock_seconds": wall_clock_seconds,
                        "wall_clock_minutes": wall_clock_minutes,
                        "tokens": tokens,

                        # 以 step 为横轴
                        "val_step/loss": val_loss,
                        "val_step/perplexity": val_ppl,

                        # 以 wall-clock time 为横轴
                        "val_time/loss": val_loss,
                        "val_time/perplexity": val_ppl,
                    },
                    step=step,
                )

        if step > 0 and step % cfg.ckpt.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=cfg.ckpt.checkpoint_path,
            )
            print(f"Saved checkpoint at iteration {step}")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=cfg.train.max_iter,
        out=cfg.ckpt.checkpoint_path,
    )
    print(f"Final checkpoint saved at iteration {cfg.train.max_iter}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
