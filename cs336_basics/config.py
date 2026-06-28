# cs336_basics/config.py

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class DataConfig:
    train_path: str | None = None
    val_path: str | None = None


@dataclass
class ModelConfig:
    vocab_size: int | None = None
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int | None = None
    rope_theta: float = 10000.0


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class LRConfig:
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    warmup_iters: int = 100
    cosine_cycle_iters: int | None = None


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_iter: int = 10000
    grad_clip: float = 1.0


@dataclass
class CheckpointConfig:
    checkpoint_path: str = "checkpoint.pt"
    resume: bool = False
    save_every: int = 1000


@dataclass
class LoggingConfig:
    log_every: int = 10
    eval_every: int = 500
    eval_batches: int = 20

    use_wandb: bool = False
    wandb_project: str = "cs336-basics"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str | None = None


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 42


@dataclass
class FullTrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr: LRConfig = field(default_factory=LRConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _load_json_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _update_dataclass_from_dict(obj: Any, values: dict[str, Any], prefix: str = "") -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            full_key = f"{prefix}.{key}" if prefix else key
            raise ValueError(f"Unknown config key: {full_key}")

        current_value = getattr(obj, key)

        if hasattr(current_value, "__dataclass_fields__") and isinstance(value, dict):
            next_prefix = f"{prefix}.{key}" if prefix else key
            _update_dataclass_from_dict(current_value, value, prefix=next_prefix)
        else:
            setattr(obj, key, value)


def _set_by_dotted_key(cfg: FullTrainConfig, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    target = cfg

    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config section: {part} in {dotted_key}")
        target = getattr(target, part)

    last_key = parts[-1]
    if not hasattr(target, last_key):
        raise ValueError(f"Unknown config key: {dotted_key}")

    setattr(target, last_key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    # tokenizer = parser.add_argument_group("tokenizer")
    # tokenizer.add_argument("--tokenizer.merge_path", type=str, default=None)
    # tokenizer.add_argument("--tokenizer.vocab_path", type=str, default=None)

    data = parser.add_argument_group("data")
    data.add_argument("--data.train_path", type=str, default=None)
    data.add_argument("--data.val_path", type=str, default=None)

    model = parser.add_argument_group("model")
    model.add_argument("--model.vocab_size", type=int, default=None)
    model.add_argument("--model.context_length", type=int, default=None)
    model.add_argument("--model.d_model", type=int, default=None)
    model.add_argument("--model.num_layers", type=int, default=None)
    model.add_argument("--model.num_heads", type=int, default=None)
    model.add_argument("--model.d_ff", type=int, default=None)
    model.add_argument("--model.rope_theta", type=float, default=None)

    optimizer = parser.add_argument_group("optimizer")
    optimizer.add_argument("--optimizer.lr", type=float, default=None)
    optimizer.add_argument("--optimizer.weight_decay", type=float, default=None)
    optimizer.add_argument("--optimizer.eps", type=float, default=None)
    optimizer.add_argument("--optimizer.beta1", type=float, default=None)
    optimizer.add_argument("--optimizer.beta2", type=float, default=None)

    lr = parser.add_argument_group("lr")
    lr.add_argument("--lr.max_learning_rate", type=float, default=None)
    lr.add_argument("--lr.min_learning_rate", type=float, default=None)
    lr.add_argument("--lr.warmup_iters", type=int, default=None)
    lr.add_argument("--lr.cosine_cycle_iters", type=int, default=None)

    train = parser.add_argument_group("train")
    train.add_argument("--train.batch_size", type=int, default=None)
    train.add_argument("--train.max_iter", type=int, default=None)
    train.add_argument("--train.grad_clip", type=float, default=None)

    ckpt = parser.add_argument_group("checkpoint")
    ckpt.add_argument("--ckpt.checkpoint_path", type=str, default=None)
    ckpt.add_argument("--ckpt.resume", action=argparse.BooleanOptionalAction, default=None)
    ckpt.add_argument("--ckpt.save_every", type=int, default=None)

    logging = parser.add_argument_group("logging")
    logging.add_argument("--logging.log_every", type=int, default=None)
    logging.add_argument("--logging.eval_every", type=int, default=None)
    logging.add_argument("--logging.eval_batches", type=int, default=None)
    logging.add_argument("--logging.use_wandb", action=argparse.BooleanOptionalAction, default=None)
    logging.add_argument("--logging.wandb_project", type=str, default=None)
    logging.add_argument("--logging.wandb_run_name", type=str, default=None)
    logging.add_argument("--logging.wandb_entity", type=str, default=None)
    logging.add_argument("--logging.wandb_mode", type=str, default=None)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--runtime.device", type=str, default=None)
    runtime.add_argument("--runtime.seed", type=int, default=None)

    return parser


def finalize_config(cfg: FullTrainConfig) -> None:
    if cfg.model.d_ff is None:
        cfg.model.d_ff = int(8 / 3 * cfg.model.d_model)

    if cfg.lr.cosine_cycle_iters is None:
        cfg.lr.cosine_cycle_iters = cfg.train.max_iter

    if cfg.runtime.device == "auto":
        cfg.runtime.device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(cfg.optimizer.betas, list):
        cfg.optimizer.betas = tuple(cfg.optimizer.betas)

    if len(cfg.optimizer.betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")


def validate_config(cfg: FullTrainConfig) -> None:
    if cfg.data.train_path is None:
        raise ValueError("data.train_path must be set")

    if cfg.model.vocab_size is None:
        raise ValueError("model.vocab_size must be set")

    if cfg.model.context_length <= 0:
        raise ValueError("model.context_length must be positive")

    if cfg.model.d_model <= 0:
        raise ValueError("model.d_model must be positive")

    if cfg.model.num_layers <= 0:
        raise ValueError("model.num_layers must be positive")

    if cfg.model.num_heads <= 0:
        raise ValueError("model.num_heads must be positive")

    if cfg.model.d_model % cfg.model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")

    if cfg.model.d_ff is None or cfg.model.d_ff <= 0:
        raise ValueError("model.d_ff must be positive")

    if cfg.train.batch_size <= 0:
        raise ValueError("train.batch_size must be positive")

    if cfg.train.max_iter <= 0:
        raise ValueError("train.max_iter must be positive")

    if cfg.train.grad_clip <= 0:
        raise ValueError("train.grad_clip must be positive")

    if cfg.lr.warmup_iters < 0:
        raise ValueError("lr.warmup_iters must be non-negative")

    if cfg.lr.cosine_cycle_iters is None or cfg.lr.cosine_cycle_iters <= 0:
        raise ValueError("lr.cosine_cycle_iters must be positive")

    if cfg.lr.min_learning_rate < 0:
        raise ValueError("lr.min_learning_rate must be non-negative")

    if cfg.lr.max_learning_rate <= 0:
        raise ValueError("lr.max_learning_rate must be positive")

    if cfg.lr.min_learning_rate > cfg.lr.max_learning_rate:
        raise ValueError("lr.min_learning_rate must be <= lr.max_learning_rate")


def parse_train_config() -> FullTrainConfig:
    parser = build_parser()
    args = parser.parse_args()

    cfg = FullTrainConfig()

    file_config = _load_json_config(args.config)
    _update_dataclass_from_dict(cfg, file_config)

    raw_args = vars(args)

    beta1 = raw_args.pop("optimizer.beta1")
    beta2 = raw_args.pop("optimizer.beta2")

    for key, value in raw_args.items():
        if key == "config":
            continue

        if value is not None:
            _set_by_dotted_key(cfg, key, value)

    if beta1 is not None or beta2 is not None:
        old_beta1, old_beta2 = cfg.optimizer.betas
        cfg.optimizer.betas = (
            beta1 if beta1 is not None else old_beta1,
            beta2 if beta2 is not None else old_beta2,
        )

    finalize_config(cfg)
    validate_config(cfg)

    return cfg