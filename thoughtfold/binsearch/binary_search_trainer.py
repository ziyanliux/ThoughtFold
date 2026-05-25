"""Binary Search DPO Trainer for efficient CoT optimization with DPO loss."""
import json
import random
from typing import cast, List

import ray
import torch
import numpy as np
from ray.actor import ActorClass
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.ray.base import AutoAcceleratorWorkers
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, ReplayBufferConfig
from xtuner.v1.ray.evaluator import EvaluatorConfig
from xtuner.v1.ray.judger import JudgerConfig
from xtuner.v1.rl.base import (
    TrainingControllerProxy,
    TrainingWorkerClass,
    TrainingWorkerProxy,
    WorkerConfig,
)
from thoughtfold.data_proto.rl_data import is_valid_for_training
from xtuner.v1.train.rl_trainer import RLTrainer, RLTrainerConfig, get_train_seq_ctx

from .binary_search_environment import BinarySearchEnvironment
from thoughtfold.rl.base.controller import TrainingController
from thoughtfold.rl.base.worker import TrainingWorker as TFTrainingWorker


class BinarySearchDPOTrainer(RLTrainer):
    """RL Trainer with Binary Search Environment and DPO loss for CoT optimization.

    This trainer extends RLTrainer to:
    1. Use BinarySearchEnvironment which generates DPO pairs during pruning
    2. Process DPO pairs in _prepare_train_data with mask-based label construction
    3. Support both Case 1 (pruned success) and Case 2 (pruned failed / overjump) DPO pairs
    """

    def __init__(
        self,
        *,
        enable_binary_search: bool = False,
        binary_search_config: dict | None = None,
        shuffle_dpo_grpo: bool = True,
        **kwargs,
    ):
        """Initialize BinarySearchDPOTrainer.

        Args:
            enable_binary_search: Enable/disable binary search optimization.
            binary_search_config: Configuration dict for binary search.
            shuffle_dpo_grpo: Whether to shuffle DPO and GRPO packs together.
        """
        self.enable_binary_search = enable_binary_search
        self.binary_search_config = binary_search_config or {}
        self.shuffle_dpo_grpo = shuffle_dpo_grpo
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        """Create BinarySearchDPOTrainer from config object."""
        self = cls(
            load_from=config.load_from,
            resources=config.resources,
            cpu_resources=config.cpu_resources,
            rollout_config=config.rollout_config,
            dataflow_config=config.dataflow_config,
            judger_config=config.judger_config,
            replay_buffer_config=config.replay_buffer_config,
            train_worker_cfg=config.train_worker_config,
            evaluator_config=config.evaluator_config,
            tokenizer_path=config.tokenizer_path,
            work_dir=config.work_dir,
            log_dir=config.log_dir,
            total_epochs=config.total_epochs,
            auto_resume=config.auto_resume,
            load_checkpoint_cfg=config.load_checkpoint_cfg,
            strict_load=config.strict_load,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_maxkeep=config.checkpoint_maxkeep,
            checkpoint_no_save_optimizer=config.checkpoint_no_save_optimizer,
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            skip_checkpoint_validation=config.skip_checkpoint_validation,
            seed=config.seed,
            debug=config.debug,
            debug_rollout=config.debug_rollout,
            rollout_steps=config.rollout_steps,
            trainer_cfg=config,
            advantage_estimator_config=config.advantage_estimator_config,
            enable_binary_search=getattr(config, 'enable_binary_search', False),
            binary_search_config=getattr(config, 'binary_search_config', None),
            shuffle_dpo_grpo=getattr(config, 'shuffle_dpo_grpo', True),
        )
        return self

    def _build_train_controller(self, train_worker_cfg: WorkerConfig) -> TrainingControllerProxy:
        """Build training controller using ThoughtFold TrainingWorker/Controller."""
        TrainingWorker = cast(
            TrainingWorkerClass,
            ray.remote(
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    }
                },
            )(TFTrainingWorker),
        )
        train_workers: list[TrainingWorkerProxy]
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(
            TrainingWorker, train_worker_cfg, self._pg)
        ray.wait([worker.ready.remote() for worker in train_workers])
        train_controller = TrainingController.remote(
            workers=train_workers, shuffle_dpo_grpo=self.shuffle_dpo_grpo)
        return train_controller

    def _build_rollout_dataflow(
        self,
        dataflow_cfg: DataFlowConfig,
        rollout_cfg: RolloutConfig,
        judger_cfg: JudgerConfig,
        replay_buffer_config: ReplayBufferConfig,
    ):
        """Build rollout dataflow with BinarySearchEnvironment."""
        env = BinarySearchEnvironment.remote(
            "thoughtfold", self._pg, rollout_cfg, self._judger_cpu_pg, judger_cfg,
            enable_binary_search=self.enable_binary_search,
            binary_search_config=self.binary_search_config,
        )
        flow = DataFlow.remote("grpo", dataflow_cfg, replay_buffer_config, env)
        return env, flow

    def _prepare_train_data(self, data_groups, pack_max_length, multimodal_train_infos=None):
        """Prepare training data with DPO pair support.

        This method extends the parent's _prepare_train_data to:
        1. Detect DPO pairs from rollout.extra_info['dpo_pair']
        2. For DPO pairs: construct (chosen, rejected) pairs with mask-based labels
        3. For non-DPO pairs: use standard GRPO logic
        """
        all_input_ids = []
        all_response_ids = []
        all_multimodal_train_infos = []
        all_routed_experts = []
        all_shifted_labels = []
        all_advantages = []
        all_rollout_logprobs = []

        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []

        is_multimodal = False
        if multimodal_train_infos and len(multimodal_train_infos) > 0:
            assert len(multimodal_train_infos) == len(data_groups), (
                f"{len(multimodal_train_infos)} vs {len(data_groups)}"
            )
            is_multimodal = True

        dpo_data_batches = []
        compression_stats_list = []

        tokenizer = self.tokenizer
        think_tag_id = tokenizer.encode("<think>", add_special_tokens=False)[0]

        for j, group in enumerate(data_groups):
            non_dpo_items = [
                data for data in group
                if not data.env.rollout.extra_info.get("dpo_pair", {}).get("is_dpo_pair", False)
            ]
            items_to_validate = non_dpo_items if non_dpo_items else group
            if not is_valid_for_training(items_to_validate):
                self.logger.error(
                    f"Skip data group due to rollout failed, empty or invalid response_ids."
                )
                continue

            multimodal_train_info = multimodal_train_infos[j] if is_multimodal else None
            if multimodal_train_info is not None and multimodal_train_info.get("pixel_values") is not None:
                pv = multimodal_train_info["pixel_values"]
                if isinstance(pv, ray.ObjectRef):
                    multimodal_train_info = dict(multimodal_train_info)
                    multimodal_train_info["pixel_values"] = ray.get(pv)

            prompt_ids = group[0].data.extra_info["train_prompt_ids"]
            dpo_skip_thinking = prompt_ids[-1] == think_tag_id

            # Collect rewards only from non-DPO samples
            reward_indices = []
            reward_values = []
            for idx, data in enumerate(group):
                rollout = data.env.rollout
                is_dpo = rollout.extra_info.get("dpo_pair", {}).get("is_dpo_pair", False)
                if not is_dpo:
                    reward_indices.append(idx)
                    reward_values.append(float(data.env.judger.reward["score"]))
                cs = rollout.extra_info.get("compression_stats")
                if cs:
                    compression_stats_list.append(cs)

            if not reward_indices:
                reward_indices = list(range(len(group)))
                reward_values = [float(data.env.judger.reward["score"]) for data in group]

            rewards = torch.tensor(reward_values, dtype=torch.float32)
            rewards_list.extend(reward_values)

            # Compute advantages for non-DPO samples
            non_dpo_group = [group[idx] for idx in reward_indices]
            if non_dpo_group and len(reward_indices) > 0:
                advantages = self._advantage_estimator.compute(rewards, non_dpo_group)
            else:
                advantages = torch.tensor([], dtype=torch.float32)

            # Map index -> advantage; DPO samples default to advantage=0
            index2advantage: dict[int, float] = {i: 0.0 for i in range(len(group))}
            for local_idx, grp_idx in enumerate(reward_indices):
                if local_idx < len(advantages):
                    index2advantage[grp_idx] = advantages[local_idx].item()

            prompt_repeat_k = len(group)
            for i in range(prompt_repeat_k):
                rollout = group[i].env.rollout
                dpo_pair_info = rollout.extra_info.pop("dpo_pair", None)

                if dpo_pair_info and dpo_pair_info.get("is_dpo_pair", False):
                    strategy = dpo_pair_info.get("strategy", "unknown")
                    if "overjump" in strategy:
                        continue

                    chosen_response = rollout.response
                    chosen_response_ids = self.tokenizer(
                        chosen_response, return_tensors="pt", add_special_tokens=False
                    )["input_ids"].flatten().tolist()

                    rejected_response_text = dpo_pair_info.get("rejected_response_text")
                    rejected_response_ids = self.tokenizer(
                        rejected_response_text, return_tensors="pt", add_special_tokens=False
                    )["input_ids"].flatten().tolist()

                    rejected_labels = dpo_pair_info.get("rejected_labels", [])
                    chosen_labels = dpo_pair_info.get("chosen_labels", [])

                    # Skip <think> tag if prompt already ends with it
                    if dpo_skip_thinking and rejected_response_ids[0] == think_tag_id:
                        rejected_response_ids = rejected_response_ids[1:]
                        rejected_labels = rejected_labels[1:]
                    if dpo_skip_thinking and chosen_response_ids[0] == think_tag_id:
                        chosen_response_ids = chosen_response_ids[1:]
                        chosen_labels = chosen_labels[1:]

                    chosen_input_ids = prompt_ids + chosen_response_ids
                    rejected_input_ids = prompt_ids + rejected_response_ids
                    chosen_shifted_labels = [-100] * (len(prompt_ids) - 1)

                    if chosen_labels and len(chosen_labels) == len(chosen_response_ids):
                        chosen_shifted_labels.extend(chosen_labels)
                    else:
                        continue
                    chosen_shifted_labels.append(-100)

                    rejected_shifted_labels = [-100] * (len(prompt_ids) - 1)
                    if rejected_labels and len(rejected_labels) == len(rejected_response_ids):
                        rejected_shifted_labels.extend(rejected_labels)
                    else:
                        continue
                    rejected_shifted_labels.append(-100)

                    assert len(chosen_shifted_labels) == len(chosen_input_ids)
                    assert len(rejected_shifted_labels) == len(rejected_input_ids)

                    # Skip if pair exceeds pack length
                    if len(chosen_input_ids) + len(rejected_input_ids) > pack_max_length:
                        continue

                    chosen_input_ids_tensor = torch.tensor(chosen_input_ids, dtype=torch.int64).unsqueeze(0)
                    chosen_shifted_labels_tensor = torch.tensor(chosen_shifted_labels, dtype=torch.int64).unsqueeze(0)
                    rejected_input_ids_tensor = torch.tensor(rejected_input_ids, dtype=torch.int64).unsqueeze(0)
                    rejected_shifted_labels_tensor = torch.tensor(rejected_shifted_labels, dtype=torch.int64).unsqueeze(0)

                    # Attach multimodal info to both chosen and rejected seq_ctx
                    chosen_seq_ctx = get_train_seq_ctx(
                        chosen_input_ids_tensor, multimodal_train_info, len(chosen_response_ids) - 1)
                    rejected_seq_ctx = get_train_seq_ctx(
                        rejected_input_ids_tensor, multimodal_train_info, len(rejected_response_ids) - 1)
                    packed_seq_ctx = SequenceContext.cat([chosen_seq_ctx, rejected_seq_ctx])
                    packed_seq_ctx.rollout_routed_experts = None
                    packed_shifted_labels = torch.cat(
                        [chosen_shifted_labels_tensor, rejected_shifted_labels_tensor], dim=1)

                    dpo_data_batches.append(dict(
                        seq_ctx=packed_seq_ctx,
                        shifted_labels=packed_shifted_labels,
                        advantage=0.0,
                        rollout_logprobs=None,
                        is_dpo_pair=True,
                        pair_type="packed",
                        chosen_len=chosen_shifted_labels_tensor.shape[1],
                        rejected_len=rejected_shifted_labels_tensor.shape[1],
                    ))
                    continue

                # ---- GRPO sample ----
                item = rollout.response
                logprobs = None
                if rollout.response_ids is not None:
                    response_ids = rollout.response_ids
                    if isinstance(response_ids, torch.Tensor):
                        response_ids = response_ids.flatten().tolist()
                    logprobs = rollout.logprobs
                    assert len(logprobs) == len(response_ids)
                    logprobs = [0] * (len(prompt_ids) - 1) + list(logprobs)
                else:
                    response_ids = self.tokenizer(
                        item, return_tensors="pt")["input_ids"].flatten().tolist()

                input_ids = prompt_ids + response_ids[:-1]
                prompt_len_list.append(len(prompt_ids))
                response_len_list.append(len(response_ids))

                advantage_scalar = index2advantage.get(i, 0.0)
                advantages_list.extend([advantage_scalar] * len(response_ids))

                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_ids
                assert len(input_ids) <= pack_max_length
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                all_input_ids.append(input_ids)
                all_response_ids.append(response_ids)
                all_shifted_labels.append(shifted_labels)
                all_advantages.append(advantage_scalar)
                all_multimodal_train_infos.append(multimodal_train_info)

                if logprobs is not None:
                    rollout_logprobs = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
                    assert rollout_logprobs.size() == shifted_labels.size()
                    all_rollout_logprobs.append(rollout_logprobs)
                else:
                    all_rollout_logprobs.append(None)

                if "routed_experts" in rollout.extra_info:
                    all_routed_experts.append(rollout.extra_info.pop("routed_experts"))
                else:
                    all_routed_experts.append(None)

        # ---- Build GRPO data_batches ----
        num_samples = len(all_input_ids)
        indices = list(range(num_samples))
        random.shuffle(indices)

        grpo_data_batches = []
        for i in indices:
            seq_ctx = get_train_seq_ctx(
                all_input_ids[i], all_multimodal_train_infos[i], len(all_response_ids[i]) - 1)
            data_dict = {
                "seq_ctx": seq_ctx,
                "shifted_labels": all_shifted_labels[i],
                "advantage": all_advantages[i],
                "rollout_logprobs": all_rollout_logprobs[i],
                "is_dpo_pair": False,
            }
            if all_routed_experts[i] is not None:
                seq_ctx.rollout_routed_experts = all_routed_experts[i]
            grpo_data_batches.append(data_dict)

        # Interleave DPO and GRPO batches
        dpo_blocks = [[b] for b in dpo_data_batches]
        non_dpo_blocks = [[b] for b in grpo_data_batches]
        blocks = dpo_blocks + non_dpo_blocks
        random.shuffle(blocks)
        data_batches = [item for blk in blocks for item in blk]

        # Compute statistics
        rewards_arr = torch.tensor(rewards_list).float() if rewards_list else torch.tensor([0.0]).float()
        advantages_arr = torch.tensor(advantages_list).float() if advantages_list else torch.tensor([0.0]).float()
        prompt_len_arr = torch.tensor(prompt_len_list).float() if prompt_len_list else torch.tensor([0.0]).float()
        response_len_arr = torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()

        info_dict = {
            "batch_size": len(rewards_list),
            "rewards/mean": rewards_arr.mean().item(),
            "rewards/min": rewards_arr.min().item(),
            "rewards/max": rewards_arr.max().item(),
            "advantages/mean": advantages_arr.mean().item(),
            "advantages/min": advantages_arr.min().item(),
            "advantages/max": advantages_arr.max().item(),
            "advantages/std": advantages_arr.std().item(),
            "advantages/pos_ratio": (advantages_arr > 0).float().mean().item(),
            "response_len/mean": response_len_arr.mean().item(),
            "response_len/min": response_len_arr.min().item(),
            "response_len/max": response_len_arr.max().item(),
            "response_len/std": response_len_arr.std().item(),
            "prompt_len/mean": prompt_len_arr.mean().item(),
            "prompt_len/min": prompt_len_arr.min().item(),
            "prompt_len/max": prompt_len_arr.max().item(),
            "dpo/pairs_count": len(dpo_data_batches),
            "dpo/grpo_count": len(grpo_data_batches),
        }

        if compression_stats_list:
            orig_arr = torch.tensor([cs["original_tokens"] for cs in compression_stats_list]).float()
            short_arr = torch.tensor([cs["shortest_tokens"] for cs in compression_stats_list]).float()
            ratio_arr = torch.tensor([cs["reduction_ratio"] for cs in compression_stats_list]).float()
            info_dict.update({
                "compression/count": len(compression_stats_list),
                "compression/original_tokens_mean": orig_arr.mean().item(),
                "compression/shortest_tokens_mean": short_arr.mean().item(),
                "compression/reduction_ratio_mean": ratio_arr.mean().item(),
                "compression/reduction_ratio_max": ratio_arr.max().item(),
                "compression/reduction_ratio_min": ratio_arr.min().item(),
            })

        self.logger.info(
            f"[DPO+GRPO] Data batch: total={len(data_batches)}, "
            f"DPO_pairs={len(dpo_data_batches)}, GRPO_samples={len(grpo_data_batches)}"
        )
        return data_batches, info_dict

    def _save_trajectories(self, data_groups, save_path):
        """Save trajectories with DPO pair filtering.

        Filters out DPO pairs when computing stats (only GRPO samples counted).
        Saves all responses (including DPO pairs) to a separate file for analysis.
        """
        rewards = []
        rollout_response_len_list = []
        version_dict = {i: 0 for i in range(self._dataflow_partial_rollout_step + 1)}
        has_dpo_pair = False

        for group in data_groups:
            group_origin = []
            for data in group:
                if data.env.rollout.extra_info.get("dpo_pair", {}).get("is_dpo_pair", False):
                    has_dpo_pair = True
                else:
                    group_origin.append(data)
            if not is_valid_for_training(group_origin):
                continue
            for data in group_origin:
                rewards.append(data.env.judger.reward["score"])
                if data.env.rollout.response_ids is not None:
                    if isinstance(data.env.rollout.response_ids, torch.Tensor):
                        response_ids = data.env.rollout.response_ids.flatten().tolist()
                    else:
                        response_ids = data.env.rollout.response_ids
                    rollout_response_len_list.append(len(response_ids))
                else:
                    response_ids = self.tokenizer.encode(
                        data.env.rollout.response, add_special_tokens=False)
                    rollout_response_len_list.append(len(response_ids))

                version = data.uid.version
                if version not in version_dict:
                    version_dict[version] = 0
                version_dict[version] += 1

        rewards_tensor = torch.tensor(rewards).float() if rewards else torch.tensor([0.0]).float()
        rollout_response_lens = torch.tensor([0.0]).float()
        if len(rollout_response_len_list) > 0:
            rollout_response_lens = torch.tensor(rollout_response_len_list).float()

        # Save all responses (including DPO pairs) to separate file
        if has_dpo_pair:
            save_path_all = save_path.parent / (save_path.name + ".all")
            with open(save_path_all, "w", encoding="utf-8") as f_all:
                for group in data_groups:
                    for data in group:
                        dpo_pair_info = data.env.rollout.extra_info.get("dpo_pair", {})
                        is_dpo_pair = dpo_pair_info.get("is_dpo_pair", False)
                        if is_dpo_pair:
                            strategy = dpo_pair_info.get("strategy", "")
                            if "overjump" in strategy:
                                continue
                        item_all = {
                            "action_id": data.uid.action_id,
                            "prompt": data.data.extra_info.get("raw_prompt", ""),
                            "response": data.env.rollout.response,
                            "versioned_response": data.env.rollout.versioned_response,
                            "is_dpo_pair": is_dpo_pair,
                        }
                        if is_dpo_pair:
                            item_all["rejected_response"] = dpo_pair_info.get("rejected_response_text", "")
                        cs = data.env.rollout.extra_info.get("compression_stats")
                        if cs:
                            item_all["compression_stats"] = cs
                        json.dump(item_all, f_all, ensure_ascii=False, indent=2)
                        f_all.write("\n")

        # Save main trajectory file (non-DPO samples only)
        _count = 0
        with open(save_path, "w", encoding="utf-8") as f:
            item = {
                "reward_mean": rewards_tensor.mean().item(),
                "reward_std": rewards_tensor.std().item(),
                "reward_max": rewards_tensor.max().item(),
                "reward_min": rewards_tensor.min().item(),
                "response_len_mean": rollout_response_lens.mean().item(),
                "response_len_std": rollout_response_lens.std().item(),
                "response_len_max": rollout_response_lens.max().item(),
                "response_len_min": rollout_response_lens.min().item(),
                "total_len": len(rewards),
                "versions": version_dict,
            }
            self.logger.info(f"versions distribution: {version_dict}")
            json.dump(item, f, ensure_ascii=False, indent=2)
            f.write("\n")

            for group in data_groups:
                group_origin = [
                    data for data in group
                    if not data.env.rollout.extra_info.get("dpo_pair", {}).get("is_dpo_pair", False)
                ]
                if not is_valid_for_training(group_origin):
                    continue
                for data in group_origin:
                    logprobs = data.env.rollout.logprobs
                    if logprobs is not None:
                        logprobs_t = logprobs if isinstance(logprobs, torch.Tensor) else torch.tensor(logprobs, dtype=torch.float32)
                        entropy = -logprobs_t.mean().item()
                    else:
                        entropy = None
                    item = {
                        "action_id": data.uid.action_id,
                        "prompt": data.data.extra_info["raw_prompt"],
                        "response": data.env.rollout.response,
                        "versioned_response": data.env.rollout.versioned_response,
                        "response_len": rollout_response_len_list[_count],
                        "origin_data_source": data.data.extra_info.get("origin_data_source", "Unknown"),
                        "versioned_response_len": data.env.rollout.versioned_num_return_tokens,
                        "label": data.data.reward_model.get("ground_truth", ""),
                        "reward": data.env.judger.reward["score"],
                        "version": data.uid.version,
                        "finish_reason": data.env.rollout.finish_reason,
                        "entropy": entropy,
                    }
                    json.dump(item, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                    _count += 1
