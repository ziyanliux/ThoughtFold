#!/usr/bin/env python
"""
ThoughtFold trainer entry point with DPO + binary search support.

This script mirrors the XTuner RLTrainer CLI while routing to
BinarySearchDPOTrainer for CoT optimization.
"""
import argparse
import importlib.util
import os

import ray
import torch.distributed as dist

from xtuner.v1.utils import Config
from .binsearch.binary_search_trainer import BinarySearchDPOTrainer


def load_config(config_path: str):
    """Load config from Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(config_module)
    return config_module


def parse_args():
    parser = argparse.ArgumentParser(description="ThoughtFold Trainer with DPO + Binary Search")
    parser.add_argument("cfg", type=str, nargs="?", help="XTuner-style config file")
    parser.add_argument("--config", type=str, default=None, help="Python config with trainer attr")
    parser.add_argument("--ray-cluster-url", type=str, default="", help="Ray cluster address")
    parser.add_argument("--work-dir", type=str, default=None, help="Override trainer.work_dir")
    parser.add_argument("--num-workers", type=int, default=None, help="Override trainer.resources.num_workers")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = args.cfg or args.config
    if cfg_path is None:
        raise SystemExit("You must provide a config: positional `cfg` or `--config`.")

    if not ray.is_initialized():
        if args.ray_cluster_url:
            ray_head_address = args.ray_cluster_url
        else:
            master_addr = os.getenv("RAY_MASTER_ADDR", "127.0.0.1")
            client_port = os.getenv("RAY_CLIENT_PORT", "10001")
            ray_head_address = f"ray://{master_addr}:{client_port}"
        ray.init(address=ray_head_address)

    config_module = load_config(cfg_path)
    trainer_cfg = config_module.trainer

    if args.work_dir is not None:
        trainer_cfg.work_dir = args.work_dir
    if args.num_workers is not None and hasattr(trainer_cfg.resources, "num_workers"):
        trainer_cfg.resources.num_workers = args.num_workers

    # Read binary search config from config module
    enable_binary_search = getattr(config_module, "enable_binary_search", False)
    binary_search_config = getattr(config_module, "binary_search_config", None)
    shuffle_dpo_grpo = getattr(config_module, "shuffle_dpo_grpo", True)

    trainer = BinarySearchDPOTrainer(
        load_from=trainer_cfg.load_from,
        resources=trainer_cfg.resources,
        cpu_resources=trainer_cfg.cpu_resources,
        rollout_config=trainer_cfg.rollout_config,
        dataflow_config=trainer_cfg.dataflow_config,
        judger_config=trainer_cfg.judger_config,
        replay_buffer_config=trainer_cfg.replay_buffer_config,
        train_worker_cfg=trainer_cfg.train_worker_config,
        evaluator_config=trainer_cfg.evaluator_config,
        tokenizer_path=trainer_cfg.tokenizer_path,
        work_dir=trainer_cfg.work_dir,
        total_epochs=trainer_cfg.total_epochs,
        hf_interval=trainer_cfg.hf_interval,
        trainer_cfg=trainer_cfg,
        enable_binary_search=enable_binary_search,
        binary_search_config=binary_search_config,
        shuffle_dpo_grpo=shuffle_dpo_grpo,
    )

    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
