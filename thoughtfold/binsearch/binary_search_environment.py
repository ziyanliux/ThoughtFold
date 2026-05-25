"""Binary Search Environment for efficient CoT optimization."""
import asyncio
import copy
import os
import re
import random
from typing import Dict, List, Optional, cast

import ray
from ray.actor import ActorClass, ActorProxy
from xtuner.v1.ray.rollout.controller import SampleParams
from thoughtfold.data_proto.rl_data import (
    RLDataFlowItem,
    RLDatasetItem,
    RLJudgerResponseItem,
    RLUIDItem,
    RLExtraDataItem,
    update_dataflow_item,
    RLRolloutResponseItem,
    is_valid_for_training,
    update_rollout_item,
    RolloutState,
)

from xtuner.v1.ray.environment.base_env import BaseEnvironment
from xtuner.v1.utils import get_logger

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert reasoner with extensive experience in all areas. "
    "You approach problems through systematic thinking and rigorous reasoning. "
    "Your response should reflect deep understanding and precise logical thinking, "
    "making your solution path and reasoning clear to others. "
    "Please put your thinking process within <think>...</think> tags."
)


class RawBinarySearchEnvironment(BaseEnvironment):
    """Environment with binary search for CoT optimization.

    Performs binary search on generated responses to find the shortest CoT
    that maintains correctness above a threshold, then applies attention-based
    fine-grained pruning.
    """

    def __init__(
        self,
        environment: str,
        rollout_pg,
        rollout_cfg=None,
        judger_pg=None,
        judger_cfg=None,
        rollout_controller=None,
        judger_controller=None,
        enable_binary_search=False,
        binary_search_config=None,
    ):
        super().__init__(
            environment, rollout_pg, rollout_cfg, judger_pg, judger_cfg,
            rollout_controller, judger_controller
        )
        worker_log_dir = rollout_cfg.worker_log_dir if rollout_cfg else judger_cfg.worker_log_dir
        self.logger = get_logger(log_dir=worker_log_dir, tag="BinarySearchEnv")
        self.rollout_cfg = rollout_cfg
        self.rollout_timeout = rollout_cfg.rollout_timeout if rollout_cfg else 1200.0
        self.judger_timeout = judger_cfg.judger_timeout if judger_cfg else 1200.0
        self.timeout_multiplier = 1.0

        # Binary search configuration
        if enable_binary_search:
            if binary_search_config is None:
                self.search_config = {
                    'repeat': 4,
                    'threshold': 0.7,
                    'max_iterations': 5,
                    'min_cot_length': 300,
                    'length_reward_weight': 1.0,
                    # Fine-grained pruning configuration
                    'enable_fine_grained_pruning': True,
                    'min_cot_length_for_pruning': 100,
                    'topk_search_min': 0.1,
                    'topk_search_max': 0.9,
                    'topk_search_iterations': 5,
                    'pruning_repeat': 4,
                    'min_valid_response_chars': 0,
                }
            else:
                self.search_config = binary_search_config

        self.enable_binary_search = enable_binary_search
        # Keywords for smart cutting
        self.KW_FWD = {"yes", "no", "so", "true", "good", "fine"}
        self.KW_BWD = {"see", "another"}

    def set_binary_search_config(self, config: dict):
        """Set binary search configuration."""
        self.search_config.update(config)
        self.enable_binary_search = config.get('enabled', True)

    async def generate(
        self,
        group_data_items: List[RLDataFlowItem],
        sample_params: Optional[SampleParams] = None,
        extra_params: Optional[Dict] = None,
    ) -> List[RLDataFlowItem]:
        """Generate responses for a batch of items using the rollout controller."""
        if self.rollout_controller:
            response_future = []
            for sample in group_data_items:
                sample.data.extra_info["root_id"] = sample.uid.root_id
                sample.data.extra_info["action_id"] = sample.uid.action_id
                rollout_extra_info = sample.data.extra_info

                fut = self.rollout_controller.rollout.remote(
                    prompt=sample.data.messages,
                    input_ids=sample.data.input_ids,
                    sample_params=sample_params,
                    extra_params=extra_params,
                    extra_info=rollout_extra_info,
                )
                response_future.append(fut)
            try:
                rollout_responses = await asyncio.wait_for(
                    asyncio.gather(*response_future),
                    timeout=self.rollout_timeout * self.timeout_multiplier
                )
            except asyncio.TimeoutError:
                self.logger.error("Rollout timeout.")
                rollout_responses = [RLRolloutResponseItem(state="skipped") for _ in group_data_items]
            group_data_items = update_rollout_item(group_data_items, rollout_responses)
        return group_data_items

    async def run(
        self, group_data_items: List[RLDataFlowItem], sample_params=None, extra_params=None
    ) -> List[RLDataFlowItem]:
        """Run full generation + judger cycle with conditional binary search.

        1. Generate responses for all samples
        2. Validate with judger to get rewards
        3. For correct samples (reward == 1): apply binary search optimization
        4. For incorrect samples: return as-is for GRPO
        """
        if extra_params is None or extra_params.get("is_eval", False) or extra_params.get("disable_routed_experts", False):
            is_eval = True
        else:
            is_eval = False

        # Step 1: Generate responses
        group_data_items = await self.generate(group_data_items, sample_params, extra_params)
        if not is_valid_for_training(group_data_items):
            return group_data_items

        # Step 2: Run judger
        if self.judger_controller:
            try:
                judger_responses: List[RLJudgerResponseItem] = await asyncio.wait_for(
                    self.judger_controller.run.remote(group_data_items),
                    timeout=self.judger_timeout * self.timeout_multiplier,
                )
            except asyncio.TimeoutError:
                self.logger.error("Judger timeout.")
                judger_responses = [
                    RLJudgerResponseItem(extra_info={"state": "failed"})
                    for _ in group_data_items
                ]
            group_data_items = update_dataflow_item(group_data_items, "env.judger", judger_responses)
        else:
            return group_data_items

        # Step 3: Conditional binary search
        if not self.enable_binary_search or not self.rollout_controller or is_eval:
            return group_data_items

        original_data = []
        for idx, item in enumerate(group_data_items):
            original_data.append({
                'idx': idx,
                'original_response': copy.deepcopy(item.env.rollout),
                'item': item
            })

        async def process_single_item(idx: int, item: RLDataFlowItem, judger_response: RLJudgerResponseItem):
            reward = judger_response.reward.get("score", 0.0)
            if reward != 1:
                # Incorrect: return for GRPO
                original_item = copy.deepcopy(item)
                original_item.env.rollout = original_data[idx]['original_response']
                original_item.env.rollout.state = RolloutState.COMPLETED
                original_item.env.judger = copy.deepcopy(judger_response)
                original_item.uid.observation_id = idx * 1000
                original_item.extra_info.extra_info["is_origin_rollout"] = True
                original_item.env.rollout.extra_info["is_origin_rollout"] = True
                return (idx, [original_item])
            else:
                # Correct: apply binary search + DPO pair generation
                original_item = copy.deepcopy(item)
                original_item.env.rollout = original_data[idx]['original_response']
                original_item.env.rollout.state = RolloutState.COMPLETED
                original_item.env.judger = copy.deepcopy(judger_response)
                original_item.uid.observation_id = idx * 1000
                original_item.extra_info.extra_info["is_origin_rollout"] = True
                original_item.env.rollout.extra_info["is_origin_rollout"] = True

                optimized_items = await self._process_with_binary_search(
                    item, original_item, sample_params, extra_params)
                result_items = [original_item]
                for opt_idx, optimized_item in enumerate(optimized_items):
                    optimized_item.env.rollout.state = RolloutState.COMPLETED
                    optimized_item.env.judger = copy.deepcopy(judger_response)
                    optimized_item.uid.observation_id = idx * 1000 + opt_idx + 1
                    optimized_item.extra_info.extra_info["is_origin_rollout"] = False
                    optimized_item.env.rollout.extra_info["is_origin_rollout"] = False
                    result_items.append(optimized_item)
                return (idx, result_items)

        results_with_idx = await asyncio.gather(*[
            process_single_item(idx, item, jr)
            for idx, (item, jr) in enumerate(zip(group_data_items, judger_responses))
        ])
        results = []
        for idx, items in sorted(results_with_idx, key=lambda x: x[0]):
            results.extend(items)
        return results

    async def _process_with_binary_search(
        self, item: RLDataFlowItem, original_item: RLDataFlowItem, sample_params, extra_params
    ) -> List[RLDataFlowItem]:
        """Process single item with binary search optimization."""
        root_id = item.uid.root_id
        full_response = item.env.rollout
        full_response.extra_info.pop("routed_experts", None)

        optimized_responses = await self._binary_search_reasoning(
            full_response, item.data.messages, sample_params, extra_params, original_item, root_id
        )
        optimized_items = []
        for opt_response in optimized_responses:
            optimized_item = copy.deepcopy(item)
            optimized_item.env.rollout = opt_response
            optimized_item.env.rollout.state = RolloutState.COMPLETED
            optimized_item.env.rollout.versioned_response_ids = []
            optimized_item.env.rollout.versioned_response = []
            optimized_item.env.rollout.versioned_logprobs = []
            optimized_item.env.rollout.versioned_num_return_tokens = []
            optimized_items.append(optimized_item)
        return optimized_items

    async def _binary_search_reasoning(
        self, full_response: RLRolloutResponseItem, prompts, sample_params,
        extra_params, original_item: RLDataFlowItem, root_id: int
    ) -> List[RLRolloutResponseItem]:
        """Phase 1: Coarse-grained binary search on CoT length.

        Performs binary search to find the shortest CoT prefix that still
        produces correct answers. Builds DPO pairs from each iteration.
        """
        full_response_len = len("<think>" + full_response.response)
        cot_content = self._extract_cot_content("<think>" + full_response.response)
        if not cot_content or len(cot_content) < self.search_config['min_cot_length']:
            return []

        tokenizer = self._get_tokenizer()
        left, right = 0, len(cot_content)
        iter_count = 0
        search_trace = []
        search_aborted = False
        best_response = copy.deepcopy(full_response)
        best_response.response = "<think>" + best_response.response
        best_chopped_cot = f"<think>{cot_content}</think>\n\n"

        while left < right and iter_count < self.search_config['max_iterations']:
            mid = (left + right) // 2
            chopped_cot_raw = self._smart_cut(cot_content, mid)
            chopped_cot = f"<think>{chopped_cot_raw}</think>\n\n"
            test_prompts, test_input_ids = self._build_test_prompts(
                prompts, chopped_cot, original_item.data.input_ids)
            validation_results = await self._validate_parallel(
                test_prompts, test_input_ids, sample_params, extra_params,
                self.search_config['repeat'], original_item
            )
            if any(r.get('aborted', False) for r in validation_results):
                search_aborted = True
                break
            iter_count += 1

            # Filter invalid validation responses
            valid_validation_results = []
            min_valid_response_chars = self.search_config.get('min_valid_response_chars', 0)
            for r in validation_results:
                resp = r['response']
                if not resp:
                    continue
                has_think = "<think>" in resp or "</think>" in resp
                length_exceed = len(resp) + len(chopped_cot) > full_response_len
                too_short = len(resp) < min_valid_response_chars
                if not has_think and not length_exceed and not too_short:
                    valid_validation_results.append(r)

            if valid_validation_results:
                filtered_correct_rate = sum(
                    1 for r in valid_validation_results if r['correct']
                ) / len(validation_results)
            else:
                filtered_correct_rate = 0.0

            candidate_is_correct = False
            if filtered_correct_rate >= self.search_config['threshold']:
                candidates = [v for v in valid_validation_results if v['correct']]
                candidate = random.choice(candidates) if candidates else valid_validation_results[0]
                candidate_resp = candidate['response']
                candidate_is_correct = True
            else:
                candidates = [v for v in validation_results if not v['correct']]
                candidate = random.choice(candidates) if candidates else validation_results[0]
                candidate_resp = candidate['response'] if candidate else ""
                candidate_is_correct = False

            combined_response_text = (chopped_cot if chopped_cot else "<think></think>\n\n") + (candidate_resp or "")
            search_trace.append({
                'iteration': iter_count,
                'cutoff_idx': mid,
                'chopped_cot': chopped_cot,
                'combined_response': combined_response_text,
                'is_correct': candidate_is_correct,
                'filtered_correct_rate': filtered_correct_rate,
            })

            if filtered_correct_rate >= self.search_config['threshold']:
                right = mid
                if mid <= self.search_config['min_cot_length']:
                    break
            else:
                left = mid + 1

        # ---- Build DPO pairs from search trace ----
        dpo_pairs: List[RLRolloutResponseItem] = []

        def _ensure_extra_info(resp: RLRolloutResponseItem):
            if not hasattr(resp, "extra_info") or resp.extra_info is None:
                resp.extra_info = {}

        for trace in search_trace:
            current_text = trace.get('combined_response', "")
            if not current_text:
                continue
            current_item = RLRolloutResponseItem(
                response=current_text,
                response_ids=None,
                num_return_tokens=0,
                finish_reason=full_response.finish_reason,
                logprobs=None,
                state=RolloutState.COMPLETED,
            )

            if trace['is_correct'] and len(current_item.response) < len(best_response.response):
                # Case 1: Pruned & correct -> chosen=current (shorter), rejected=best (longer)
                rejected_cot = self._extract_cot_content(best_response.response)
                chosen_cot = self._extract_cot_content(current_item.response)
                if not rejected_cot or not chosen_cot:
                    best_response = current_item
                    continue

                rejected_full_tokens = tokenizer.encode(best_response.response, add_special_tokens=False)
                chosen_full_tokens = tokenizer.encode(current_item.response, add_special_tokens=False)
                think_tag_len = len(tokenizer.encode("<think>", add_special_tokens=False))
                rejected_cot_len = len(tokenizer.encode(rejected_cot, add_special_tokens=False))
                chosen_cot_len = len(tokenizer.encode(chosen_cot, add_special_tokens=False))
                pruned_len = rejected_cot_len - chosen_cot_len
                if pruned_len <= 0:
                    best_response = current_item
                    continue

                # Handle BPE boundary effect for <think> tag
                chosen_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + chosen_cot, add_special_tokens=False)) == think_tag_len + chosen_cot_len - 1:
                    chosen_think_tag_len = think_tag_len - 1
                rejected_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + rejected_cot, add_special_tokens=False)) == think_tag_len + rejected_cot_len - 1:
                    rejected_think_tag_len = think_tag_len - 1

                # Rejected: only penalize the pruned reasoning portion
                rejected_labels = [-100] * len(rejected_full_tokens)
                start_idx = rejected_think_tag_len + chosen_cot_len
                for i in range(start_idx, min(start_idx + pruned_len, len(rejected_labels))):
                    rejected_labels[i] = rejected_full_tokens[i]

                # Chosen: mask reasoning, keep answer
                chosen_labels = list(chosen_full_tokens)
                for i in range(min(chosen_think_tag_len + chosen_cot_len, len(chosen_labels))):
                    chosen_labels[i] = -100

                if rejected_labels and chosen_labels:
                    rejected_loss_tokens = sum(1 for l in rejected_labels if l != -100)
                    chosen_loss_tokens = sum(1 for l in chosen_labels if l != -100)
                    self.logger.info(
                        f"[DPO-BinSearch] Group[{root_id}] iter={trace['iteration']} cutoff={trace['cutoff_idx']} | "
                        f"rejected={len(rejected_full_tokens)}t(loss={rejected_loss_tokens}, pruned={pruned_len}) "
                        f"chosen={len(chosen_full_tokens)}t(loss={chosen_loss_tokens})"
                    )
                    chosen_item = copy.deepcopy(current_item)
                    _ensure_extra_info(chosen_item)
                    chosen_item.extra_info['dpo_pair'] = {
                        'is_dpo_pair': True,
                        'strategy': 'binary_search',
                        'rejected_response_text': best_response.response,
                        'rejected_response_ids': best_response.response_ids,
                        'rejected_labels': rejected_labels,
                        'chosen_labels': chosen_labels,
                    }
                    dpo_pairs.append(chosen_item)
                    best_response = current_item
                    best_chopped_cot = trace['chopped_cot']
                continue

            if not trace['is_correct']:
                # Case 2: Overjump -> rejected=current (too short), chosen=best (correct)
                rejected_full_tokens = tokenizer.encode(current_item.response, add_special_tokens=False)
                chosen_full_tokens = tokenizer.encode(best_response.response, add_special_tokens=False)
                think_tag_len = len(tokenizer.encode("<think>", add_special_tokens=False))
                rejected_cot = self._extract_cot_content(current_item.response)
                rejected_cot_len = len(tokenizer.encode(rejected_cot, add_special_tokens=False)) if rejected_cot else 0
                chosen_cot = self._extract_cot_content(best_response.response)
                chosen_cot_len = len(tokenizer.encode(chosen_cot, add_special_tokens=False)) if chosen_cot else 0

                if not rejected_cot or not chosen_cot:
                    continue

                # Handle BPE boundary
                chosen_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + chosen_cot, add_special_tokens=False)) == think_tag_len + chosen_cot_len - 1:
                    chosen_think_tag_len = think_tag_len - 1
                rejected_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + rejected_cot, add_special_tokens=False)) == think_tag_len + rejected_cot_len - 1:
                    rejected_think_tag_len = think_tag_len - 1

                # Both sides: mask reasoning, keep answer only
                rejected_labels = list(rejected_full_tokens)
                chosen_labels = list(chosen_full_tokens)
                for i in range(rejected_think_tag_len, min(rejected_think_tag_len + rejected_cot_len, len(rejected_labels))):
                    rejected_labels[i] = -100
                for i in range(chosen_think_tag_len, min(chosen_think_tag_len + chosen_cot_len, len(chosen_labels))):
                    chosen_labels[i] = -100

                rejected_loss_tokens = sum(1 for l in rejected_labels if l != -100)
                chosen_loss_tokens = sum(1 for l in chosen_labels if l != -100)
                self.logger.info(
                    f"[DPO-BinSearch] Group[{root_id}] overjump iter={trace['iteration']} cutoff={trace['cutoff_idx']} "
                    f"rate={trace['filtered_correct_rate']:.2f} | "
                    f"rejected={len(rejected_full_tokens)}t(loss={rejected_loss_tokens}) "
                    f"chosen={len(chosen_full_tokens)}t(loss={chosen_loss_tokens})"
                )
                chosen_item = copy.deepcopy(best_response)
                _ensure_extra_info(chosen_item)
                chosen_item.extra_info['dpo_pair'] = {
                    'is_dpo_pair': True,
                    'strategy': 'binary_search_overjump',
                    'rejected_response_text': current_item.response,
                    'rejected_response_ids': current_item.response_ids,
                    'rejected_labels': rejected_labels,
                    'chosen_labels': chosen_labels,
                }
                dpo_pairs.append(chosen_item)

        # ---- Phase 2: Fine-grained pruning (attention-based) ----
        if self.search_config.get('enable_fine_grained_pruning', True) and not search_aborted:
            try:
                cot_for_best = self._extract_cot_content(best_chopped_cot)
                if cot_for_best and len(cot_for_best.strip()) > 0:
                    attn_extra_params = copy.deepcopy(extra_params) if extra_params else {}
                    attn_extra_params['attn'] = True
                    attn_extra_params['attn_only'] = True
                    attn_extra_params['attn_response_text'] = best_response.response

                    attn_extra_info = copy.deepcopy(original_item.data.extra_info) if hasattr(original_item.data, 'extra_info') else {}
                    attn_response = await self.rollout_controller.rollout.remote(
                        prompt=prompts,
                        input_ids=original_item.data.input_ids,
                        sample_params=sample_params,
                        extra_params=attn_extra_params,
                        extra_info=attn_extra_info,
                        session_id=root_id
                    )
                    if attn_response.attention_scores is not None:
                        best_response.attention_scores = attn_response.attention_scores
                        best_response.reasoning_token_ids = attn_response.reasoning_token_ids

                        pruning_extra_params = copy.deepcopy(extra_params) if extra_params else {}
                        pruning_extra_params.pop('attn', None)
                        pruning_extra_params.pop('attn_only', None)
                        pruning_extra_params.pop('attn_response_text', None)

                        fg_pairs = await self._fine_grained_pruning(
                            shortest_response=best_response,
                            shortest_chopped_cot=best_chopped_cot,
                            prompts=prompts,
                            sample_params=sample_params,
                            extra_params=pruning_extra_params,
                            original_item=original_item,
                            root_id=root_id
                        )
                        dpo_pairs.extend(fg_pairs)
            except Exception as e:
                self.logger.info(f"Group[{root_id}] Fine-grained failed: {e}")
        return dpo_pairs

    async def _validate_parallel(
        self, test_prompts, test_input_ids, sample_params, extra_params,
        repeat: int, original_item: RLDataFlowItem
    ):
        """Validate pruned CoT with parallel sampling + batch judging."""
        rollout_extra_info = copy.deepcopy(original_item.data.extra_info) if hasattr(original_item.data, 'extra_info') else {}

        async def validate_single(index: int):
            try:
                response = await self.rollout_controller.rollout.remote(
                    prompt=test_prompts,
                    input_ids=test_input_ids,
                    sample_params=sample_params,
                    extra_params=extra_params,
                    extra_info=rollout_extra_info,
                )
                finish_reason = response.finish_reason
                is_aborted = finish_reason in ("abort", "aborted", "skipped")
                base_uid = original_item.uid
                data_item = RLDataFlowItem(
                    uid=RLUIDItem(
                        env=base_uid.env,
                        root_id=base_uid.root_id,
                        action_id=base_uid.action_id,
                        observation_id=index,
                    ),
                    data=RLDatasetItem(
                        messages=test_prompts,
                        reward_model=original_item.data.reward_model,
                        data_source=original_item.data.data_source,
                        extra_info=original_item.data.extra_info,
                    ),
                    extra_info=RLExtraDataItem()
                )
                data_item.env.rollout.response = response.response
                data_item.env.rollout.response_ids = response.response_ids
                data_item.env.rollout.finish_reason = finish_reason
                data_item.env.rollout.logprobs = response.logprobs
                return {
                    'index': index,
                    'data_item': data_item,
                    'response': response.response,
                    'response_ids': response.response_ids,
                    'logprobs': response.logprobs,
                    'finish_reason': finish_reason,
                    'aborted': is_aborted,
                }
            except Exception as e:
                self.logger.info(f"Validation failed for index {index}: {e}")
                return {
                    'index': index, 'data_item': None,
                    'response': "", 'response_ids': None,
                    'logprobs': None, 'finish_reason': "failed", 'aborted': True,
                }

        rollout_results = await asyncio.gather(*[validate_single(i) for i in range(repeat)])

        # Batch judger call
        data_items_for_judger = []
        valid_results = []
        for result in rollout_results:
            if result['data_item'] is not None:
                data_items_for_judger.append(result['data_item'])
                valid_results.append(result)
            else:
                valid_results.append({**result, 'correct': False, 'reward': {"score": 0.0}})

        if self.judger_controller and len(data_items_for_judger) > 0:
            try:
                judger_responses = await self.judger_controller.run.remote(data_items_for_judger)
                judger_idx = 0
                for result in valid_results:
                    if result['data_item'] is not None:
                        matched_response = judger_responses[judger_idx]
                        reward = matched_response.reward
                        result['reward'] = reward
                        result['correct'] = reward.get("score", 0.0) > 0.0
                        judger_idx += 1
            except Exception as e:
                raise AssertionError(f"Judger call failed: {e}") from e
        else:
            for result in valid_results:
                if result['data_item'] is not None:
                    result['correct'] = bool(result['response'].strip())
                    result['reward'] = {"score": 1.0 if result['correct'] else 0.0}

        return [{
            'index': r['index'],
            'response': r['response'],
            'response_ids': r['response_ids'],
            'logprobs': r['logprobs'],
            'correct': r.get('correct', False),
            'finish_reason': r['finish_reason'],
            'reward': r.get('reward', {"score": 0.0}),
            'aborted': r.get('aborted', False),
        } for r in valid_results]

    async def _fine_grained_pruning(
        self, shortest_response: RLRolloutResponseItem, shortest_chopped_cot: str,
        prompts: List, sample_params, extra_params, original_item: RLDataFlowItem, root_id: int
    ) -> List[RLRolloutResponseItem]:
        """Phase 2: Fine-grained pruning using attention scores.

        Binary search on attention-weighted sentence retention ratio (topk_ratio).
        Builds DPO pairs from each iteration:
        - Case 1: shorter & correct -> best=rejected, current=chosen
        - Case 2: shorter & incorrect -> current=rejected, best=chosen
        """
        tokenizer = self._get_tokenizer()
        cot_content = self._extract_cot_content(shortest_chopped_cot)
        if not cot_content:
            return []
        min_cot_length = self.search_config.get('min_cot_length_for_pruning', 100)
        if len(cot_content) < min_cot_length:
            return []

        shortest_cot_with_tags = self._extract_cot_with_tags(shortest_response.response)
        attention_scores = shortest_response.attention_scores
        reasoning_token_ids = getattr(shortest_response, 'reasoning_token_ids', None)

        if not attention_scores or len(attention_scores) == 0:
            return []
        if reasoning_token_ids is None or len(reasoning_token_ids) == 0:
            return []
        assert len(attention_scores) == len(reasoning_token_ids), (
            f"Attention scores length ({len(attention_scores)}) != reasoning_token_ids length ({len(reasoning_token_ids)})"
        )

        topk_min = self.search_config.get('topk_search_min', 0.1)
        topk_max = self.search_config.get('topk_search_max', 0.9)
        max_iterations = self.search_config.get('topk_search_iterations', 5)
        left, right = topk_min, topk_max
        iter_count = 0
        pruning_trace: List[dict] = []

        sentences, _ = self._split_cot_into_sentences(cot_content)
        if len(sentences) <= 1:
            return []

        while left < right and iter_count < max_iterations:
            mid_topk = (left + right) / 2.0
            iter_count += 1
            pruned_cot, num_origin, num_keep, pruning_mask_indices, token_to_sentence, sent_prune_mask = \
                self._prune_reasoning_by_attention(
                    cot_content=cot_content,
                    attention_scores=attention_scores,
                    reasoning_token_ids=reasoning_token_ids,
                    topk_ratio=mid_topk,
                    root_id=root_id
                )
            if not pruned_cot or len(pruned_cot) < 30:
                left = mid_topk + 0.01
                continue
            if len(pruned_cot) < min_cot_length:
                break

            pruned_cot_with_tags = f"<think>{pruned_cot}</think>\n\n"
            pruned_test_prompts, pruned_test_input_ids = self._build_test_prompts(
                prompts, pruned_cot_with_tags, original_item.data.input_ids)
            validation_results = await self._validate_parallel(
                pruned_test_prompts, pruned_test_input_ids, sample_params, extra_params,
                self.search_config.get('pruning_repeat', 4), original_item
            )
            if any(r.get('aborted', False) for r in validation_results):
                break

            full_response_len = len(shortest_response.response)
            pruned_cot_len = len(pruned_cot_with_tags)
            valid_validation_results = []
            min_valid_response_chars = self.search_config.get('min_valid_response_chars', 0)
            for r in validation_results:
                resp = r['response']
                if not resp:
                    continue
                has_think = "<think>" in resp or "</think>" in resp
                length_exceed = len(resp) + pruned_cot_len > full_response_len
                too_short = len(resp) < min_valid_response_chars
                if not has_think and not length_exceed and not too_short:
                    valid_validation_results.append(r)

            correct_validations = [r for r in valid_validation_results if r['correct']]
            filtered_correct_rate = len(correct_validations) / len(validation_results) if valid_validation_results else 0.0
            threshold = self.search_config['threshold']

            if filtered_correct_rate >= threshold:
                candidate = random.choice(correct_validations) if correct_validations else correct_validations[0]
                candidate_resp = candidate['response']
                candidate_is_correct = True
            else:
                candidates = [v for v in validation_results if not v['correct']]
                candidate = random.choice(candidates) if candidates else validation_results[0]
                candidate_resp = candidate['response']
                candidate_is_correct = False

            combined_response_text = pruned_cot_with_tags + (candidate_resp or "")
            pruning_trace.append({
                'iteration': iter_count,
                'mid_topk': mid_topk,
                'pruned_cot_with_tags': pruned_cot_with_tags,
                'combined_response': combined_response_text,
                'is_correct': candidate_is_correct,
                'filtered_correct_rate': filtered_correct_rate,
                'pruning_mask_indices': pruning_mask_indices,
                'sent_prune_mask': sent_prune_mask,
                'token_to_sentence': token_to_sentence,
            })

            self.logger.info(
                f"Group[{root_id}] Fine-grained iter {iter_count}: topk={mid_topk:.2f}, "
                f"{num_origin}->{num_keep} steps, correct_rate={filtered_correct_rate:.2f}"
            )
            if filtered_correct_rate >= threshold:
                if len(pruned_cot) <= min_cot_length or num_keep <= 1:
                    break
                right = mid_topk
            else:
                left = mid_topk + 0.01

        # ---- Backtrack phase: build DPO pairs ----
        dpo_pairs: List[RLRolloutResponseItem] = []
        best_response = copy.deepcopy(shortest_response)
        best_pruning_mask = [0] * len(reasoning_token_ids)
        best_sent_prune_mask = []
        if pruning_trace and 'sent_prune_mask' in pruning_trace[0]:
            best_sent_prune_mask = [0] * len(pruning_trace[0]['sent_prune_mask'])

        def _ensure_extra_info(resp: RLRolloutResponseItem):
            if not hasattr(resp, "extra_info") or resp.extra_info is None:
                resp.extra_info = {}

        for trace in pruning_trace:
            current_text = trace.get('combined_response', "")
            if not current_text:
                continue
            if current_text.count('<think>') > 1:
                continue
            current_item = RLRolloutResponseItem(
                response=current_text, response_ids=None, num_return_tokens=0,
                finish_reason=shortest_response.finish_reason, logprobs=None,
                state=RolloutState.COMPLETED,
            )

            # Case 1: shorter & correct
            if trace['is_correct'] and len(current_item.response) < len(best_response.response):
                rejected_cot = self._extract_cot_content(best_response.response)
                chosen_cot = self._extract_cot_content(current_item.response)
                chosen_pruning_mask = trace.get('pruning_mask_indices', [])
                if not rejected_cot or not chosen_cot:
                    best_response = current_item
                    best_pruning_mask = chosen_pruning_mask
                    continue

                rejected_full_tokens = tokenizer.encode(best_response.response, add_special_tokens=False)
                chosen_full_tokens = tokenizer.encode(current_item.response, add_special_tokens=False)
                think_tag_len = len(tokenizer.encode("<think>", add_special_tokens=False))
                rejected_cot_len = len(tokenizer.encode(rejected_cot, add_special_tokens=False))
                chosen_cot_len = len(tokenizer.encode(chosen_cot, add_special_tokens=False))
                pruned_len = rejected_cot_len - chosen_cot_len
                if pruned_len <= 0:
                    best_response = current_item
                    best_pruning_mask = chosen_pruning_mask
                    continue

                # BPE boundary handling
                chosen_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + chosen_cot, add_special_tokens=False)) == think_tag_len + chosen_cot_len - 1:
                    chosen_think_tag_len = think_tag_len - 1
                rejected_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + rejected_cot, add_special_tokens=False)) == think_tag_len + rejected_cot_len - 1:
                    rejected_think_tag_len = think_tag_len - 1

                # Build rejected labels: penalize only pruned sentences
                rejected_labels = list(rejected_full_tokens)
                start = rejected_think_tag_len
                chosen_sent_prune_mask = trace.get('sent_prune_mask', [])
                penalty_sent_indices = set()
                reward_sent_indices = set()
                if best_sent_prune_mask and chosen_sent_prune_mask and len(best_sent_prune_mask) == len(chosen_sent_prune_mask):
                    for sent_idx in range(len(best_sent_prune_mask)):
                        if best_sent_prune_mask[sent_idx] == 0 and chosen_sent_prune_mask[sent_idx] == 1:
                            penalty_sent_indices.add(sent_idx)
                        if chosen_sent_prune_mask[sent_idx] == 0 and sent_idx > 0:
                            if (sent_idx - 1) in penalty_sent_indices:
                                reward_sent_indices.add(sent_idx)

                # Mask <think> tag
                for i in range(start):
                    rejected_labels[i] = -100

                rejected_t2s = self._build_token_to_sentence(tokenizer, sentences, best_sent_prune_mask)
                if penalty_sent_indices and rejected_t2s:
                    for j in range(rejected_cot_len):
                        idx = start + j
                        if idx < len(rejected_labels) and j < len(rejected_t2s):
                            if rejected_t2s[j] not in penalty_sent_indices:
                                rejected_labels[idx] = -100
                        else:
                            rejected_labels[idx] = -100
                else:
                    for i in range(start, min(start + rejected_cot_len, len(rejected_labels))):
                        rejected_labels[i] = -100
                start += rejected_cot_len
                for i in range(start, len(rejected_labels)):
                    rejected_labels[i] = -100

                # Build chosen labels: reward bridging sentences
                chosen_labels = list(chosen_full_tokens)
                chosen_start = chosen_think_tag_len
                for i in range(chosen_start):
                    chosen_labels[i] = -100
                chosen_t2s = self._build_token_to_sentence(tokenizer, sentences, chosen_sent_prune_mask)
                if reward_sent_indices and chosen_t2s:
                    for j in range(chosen_cot_len):
                        idx = chosen_start + j
                        if idx < len(chosen_labels) and j < len(chosen_t2s):
                            if chosen_t2s[j] not in reward_sent_indices:
                                chosen_labels[idx] = -100
                        elif idx < len(chosen_labels):
                            chosen_labels[idx] = -100
                chosen_start += chosen_cot_len

                if rejected_labels and chosen_labels:
                    rejected_loss_tokens = sum(1 for l in rejected_labels if l != -100)
                    chosen_loss_tokens = sum(1 for l in chosen_labels if l != -100)
                    self.logger.info(
                        f"[DPO-FineGrained] Group[{root_id}] iter={trace['iteration']} topk={trace['mid_topk']:.2f} | "
                        f"rejected={len(rejected_full_tokens)}t(loss={rejected_loss_tokens}) "
                        f"chosen={len(chosen_full_tokens)}t(loss={chosen_loss_tokens})"
                    )
                    chosen_item = copy.deepcopy(current_item)
                    _ensure_extra_info(chosen_item)
                    chosen_item.extra_info['dpo_pair'] = {
                        'is_dpo_pair': True,
                        'strategy': 'fine_grained_pruning',
                        'rejected_response_text': best_response.response,
                        'rejected_response_ids': best_response.response_ids,
                        'rejected_labels': rejected_labels,
                        'chosen_labels': chosen_labels,
                    }
                    dpo_pairs.append(chosen_item)
                best_response = current_item
                best_pruning_mask = chosen_pruning_mask
                if chosen_sent_prune_mask:
                    best_sent_prune_mask = chosen_sent_prune_mask
                continue

            # Case 2: overjump (incorrect)
            if not trace['is_correct']:
                rejected_cot = self._extract_cot_content(current_item.response)
                chosen_cot = self._extract_cot_content(best_response.response)
                if not rejected_cot or not chosen_cot:
                    continue
                rejected_full_tokens = tokenizer.encode(current_item.response, add_special_tokens=False)
                chosen_full_tokens = tokenizer.encode(best_response.response, add_special_tokens=False)
                think_tag_len = len(tokenizer.encode("<think>", add_special_tokens=False))
                rejected_cot_len = len(tokenizer.encode(rejected_cot, add_special_tokens=False))
                chosen_cot_len = len(tokenizer.encode(chosen_cot, add_special_tokens=False))

                # BPE boundary
                chosen_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + chosen_cot, add_special_tokens=False)) == think_tag_len + chosen_cot_len - 1:
                    chosen_think_tag_len = think_tag_len - 1
                rejected_think_tag_len = think_tag_len
                if len(tokenizer.encode("<think>" + rejected_cot, add_special_tokens=False)) == think_tag_len + rejected_cot_len - 1:
                    rejected_think_tag_len = think_tag_len - 1

                # Find overjump sentences
                rejected_sent_prune_mask = trace.get('sent_prune_mask', [])
                overjump_sent_indices = set()
                penalty_sent_indices = set()
                if rejected_sent_prune_mask and best_sent_prune_mask and len(rejected_sent_prune_mask) == len(best_sent_prune_mask):
                    for sent_idx in range(len(rejected_sent_prune_mask)):
                        if rejected_sent_prune_mask[sent_idx] == 1 and best_sent_prune_mask[sent_idx] == 0:
                            overjump_sent_indices.add(sent_idx)
                        if rejected_sent_prune_mask[sent_idx] == 0 and sent_idx > 0:
                            if (sent_idx - 1) in overjump_sent_indices:
                                penalty_sent_indices.add(sent_idx)

                # Rejected: penalize bridging sentences after overjumped ones
                rejected_labels = [-100] * len(rejected_full_tokens)
                rejected_start = rejected_think_tag_len
                rejected_t2s = self._build_token_to_sentence(tokenizer, sentences, rejected_sent_prune_mask)
                if penalty_sent_indices and rejected_t2s:
                    for j in range(rejected_cot_len):
                        idx = rejected_start + j
                        if idx < len(rejected_labels) and j < len(rejected_t2s):
                            if rejected_t2s[j] in penalty_sent_indices:
                                rejected_labels[idx] = rejected_full_tokens[idx]
                rejected_start += rejected_cot_len
                for i in range(rejected_start + 1, len(rejected_labels)):
                    rejected_labels[i] = rejected_full_tokens[i]

                # Chosen: only answer part
                chosen_labels = [-100] * len(chosen_full_tokens)
                chosen_start = chosen_think_tag_len + chosen_cot_len
                if chosen_start < len(chosen_labels):
                    for i in range(chosen_start, len(chosen_labels)):
                        chosen_labels[i] = chosen_full_tokens[i]

                if rejected_labels and chosen_labels:
                    rejected_loss_tokens = sum(1 for l in rejected_labels if l != -100)
                    chosen_loss_tokens = sum(1 for l in chosen_labels if l != -100)
                    self.logger.info(
                        f"[DPO-FineGrained] Group[{root_id}] overjump iter={trace['iteration']} topk={trace['mid_topk']:.2f} "
                        f"rate={trace['filtered_correct_rate']:.2f} | "
                        f"rejected={len(rejected_full_tokens)}t(loss={rejected_loss_tokens}) "
                        f"chosen={len(chosen_full_tokens)}t(loss={chosen_loss_tokens})"
                    )
                    chosen_item = copy.deepcopy(best_response)
                    _ensure_extra_info(chosen_item)
                    chosen_item.extra_info['dpo_pair'] = {
                        'is_dpo_pair': True,
                        'strategy': 'fine_grained_overjump',
                        'rejected_response_text': current_item.response,
                        'rejected_response_ids': current_item.response_ids,
                        'rejected_labels': rejected_labels,
                        'chosen_labels': chosen_labels,
                    }
                    dpo_pairs.append(chosen_item)
                continue
        return dpo_pairs

    def _prune_reasoning_by_attention(
        self, cot_content: str, attention_scores: List[float],
        reasoning_token_ids: List[int], topk_ratio: float, root_id: int
    ) -> tuple:
        """Prune reasoning at sentence level based on attention scores.

        Returns:
            (pruned_cot, num_steps, num_keep, pruning_mask, token_to_sentence, sent_prune_mask)
        """
        if not cot_content or len(attention_scores) == 0:
            return "", 0, 0, [0] * len(reasoning_token_ids), [-1] * len(reasoning_token_ids), []

        attn_tokenizer = self._get_attention_tokenizer()
        if attn_tokenizer is None:
            return cot_content, 0, 0, [0] * len(reasoning_token_ids), [-1] * len(reasoning_token_ids), []

        sentences, separators = self._split_cot_into_sentences(cot_content)
        if not sentences:
            return cot_content, 0, 0, [0] * len(reasoning_token_ids), [-1] * len(reasoning_token_ids), [0] * len(sentences)

        # Map tokens to sentences
        sentence_data = []
        token_to_sentence = []
        token_start = 0
        for sent_idx, sent_text in enumerate(sentences):
            sent_token_len = len(attn_tokenizer.encode(sent_text, add_special_tokens=False))
            token_end = token_start + sent_token_len
            sentence_data.append({
                'index': sent_idx,
                'text': sent_text,
                'token_start': token_start,
                'token_end': token_end,
                'token_len': sent_token_len
            })
            token_to_sentence.extend([sent_idx] * sent_token_len)
            token_start = token_end

        assert len(token_to_sentence) == len(reasoning_token_ids), (
            f"Token count mismatch: token_to_sentence={len(token_to_sentence)}, "
            f"reasoning_token_ids={len(reasoning_token_ids)}."
        )

        # Compute mean attention score per sentence
        for sent_data in sentence_data:
            t_start = sent_data['token_start']
            t_end = sent_data['token_end']
            scores = [attention_scores[t] for t in range(t_start, min(t_end, len(attention_scores)))]
            sent_data['score'] = (sum(scores) / len(scores)) if scores else 0.0

        # Select top-k sentences by attention score
        num_keep = max(1, int(len(sentence_data) * topk_ratio))
        sorted_by_score = sorted(sentence_data, key=lambda x: x['score'], reverse=True)
        keep_indices = set(s['index'] for s in sorted_by_score[:num_keep])

        # Generate sentence-level prune mask
        sent_prune_mask = [1 if i not in keep_indices else 0 for i in range(len(sentences))]

        # Generate token-level pruning mask
        pruning_mask = [0] * len(reasoning_token_ids)
        for sent_data in sentence_data:
            if sent_data['index'] not in keep_indices:
                for t_idx in range(sent_data['token_start'], sent_data['token_end']):
                    if t_idx < len(pruning_mask):
                        pruning_mask[t_idx] = 1

        # Reconstruct pruned text
        pruned_cot = ""
        kept_sent_indices = sorted(i for i in range(len(sentences)) if i in keep_indices)
        for i, sent_idx in enumerate(kept_sent_indices):
            pruned_cot += sentences[sent_idx]
            if i < len(kept_sent_indices) - 1 and sent_idx < len(separators):
                if separators[sent_idx]:
                    pruned_cot += separators[sent_idx]

        return pruned_cot, len(sentence_data), len(keep_indices), pruning_mask, token_to_sentence, sent_prune_mask

    # ---- Helper methods ----

    def _get_tokenizer(self):
        """Lazy load tokenizer."""
        if not hasattr(self, '_tokenizer') or self._tokenizer is None:
            if not self.rollout_cfg or not hasattr(self.rollout_cfg, 'model_path'):
                return None
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.rollout_cfg.model_path, trust_remote_code=True)
        return self._tokenizer

    def _get_attention_tokenizer(self):
        """Lazy load attention model tokenizer."""
        if not hasattr(self, '_attn_tokenizer') or self._attn_tokenizer is None:
            extra = getattr(self.rollout_cfg, 'extra_rollout_config', None) or {}
            attn_path = extra.get('attention_model_path')
            if not attn_path:
                return self._get_tokenizer()
            try:
                from transformers import AutoTokenizer
                self._attn_tokenizer = AutoTokenizer.from_pretrained(attn_path, trust_remote_code=True)
            except Exception:
                return self._get_tokenizer()
        return self._attn_tokenizer

    def _build_token_to_sentence(self, tokenizer, sentences, sent_mask=None):
        """Build token_to_sentence mapping."""
        token_to_sentence = []
        for sent_idx, sent_text in enumerate(sentences):
            if sent_mask is not None and sent_idx < len(sent_mask) and sent_mask[sent_idx] != 0:
                continue
            sent_token_len = len(tokenizer.encode(sent_text, add_special_tokens=False))
            token_to_sentence.extend([sent_idx] * sent_token_len)
        return token_to_sentence

    def _smart_cut(self, cot: str, mid: int) -> str:
        """Cut CoT at a sentence boundary near `mid`."""
        chopped = cot[:mid]
        mid = len(chopped)
        if not chopped.endswith('\n'):
            last_nl = chopped.rfind('\n')
            if last_nl == -1:
                return ""
            chopped = chopped[:last_nl + 1]
            mid = len(chopped)
        if not chopped:
            return ""
        prev1 = chopped.rfind("\n", 0, mid)
        prev2 = chopped.rfind("\n", 0, prev1)
        window_start = (prev2 + 1) if prev2 != -1 else 0
        window = chopped[window_start:mid]
        toks = {re.sub(r"[^\w]", "", t.lower()) for t in window.split()}
        has_fwd = bool(toks & self.KW_FWD)
        has_bwd = bool(toks & self.KW_BWD)
        if has_bwd:
            cut = prev1 if prev1 != -1 else mid
        elif has_fwd:
            nxt = chopped.find("\n", mid)
            cut = (nxt + 1) if nxt != -1 else len(chopped)
        else:
            cut = mid
        return chopped[:cut]

    def _extract_cot_content(self, response: str) -> str:
        """Extract content between <think>...</think> tags."""
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, response, re.DOTALL)
        if matches:
            return matches[0]
        truncated_pattern = r'<think>(.*)'
        truncated_matches = re.findall(truncated_pattern, response, re.DOTALL)
        if truncated_matches:
            return truncated_matches[0]
        return ""

    def _extract_cot_with_tags(self, response: str) -> str:
        """Extract full <think>...</think> block including tags."""
        think_pattern = r'<think>.*?</think>'
        matches = re.findall(think_pattern, response, re.DOTALL)
        if matches:
            return matches[0]
        truncated_pattern = r'<think>.*'
        truncated_matches = re.findall(truncated_pattern, response, re.DOTALL)
        if truncated_matches:
            return truncated_matches[0]
        return ""

    def _split_cot_into_sentences(self, text: str) -> tuple:
        """Split CoT text into reasoning steps by '\\n\\n' delimiter.

        Returns:
            (steps, separators): steps include trailing separator for reconstruction.
        """
        if not text:
            return [], []
        parts = text.split('\n\n')
        combined_sentences = []
        for i, part in enumerate(parts):
            if not part:
                continue
            if i == len(parts) - 1:
                combined_sentences.append(part)
            else:
                combined_sentences.append(part + '\n\n')
        return combined_sentences, [''] * len(combined_sentences)

    def _build_test_prompts(self, original_prompts: List, chopped_cot: str, original_input_ids=None) -> tuple:
        """Build test prompts with chopped CoT as assistant prefix."""
        test_prompts = copy.deepcopy(original_prompts)

        found_assistant = False
        for prompt in test_prompts:
            if prompt.get("role") == "assistant":
                prompt["content"] = chopped_cot
                found_assistant = True
                break
        if not found_assistant:
            test_prompts.append({"role": "assistant", "content": chopped_cot})

        input_ids = None
        if original_input_ids is not None:
            tokenizer = self._get_tokenizer()
            if tokenizer is not None:
                try:
                    text = tokenizer.apply_chat_template(test_prompts, tokenize=False, add_generation_prompt=True)
                    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                except Exception:
                    pass
        return test_prompts, input_ids


BinarySearchEnvironment = cast(
    ActorClass[RawBinarySearchEnvironment],
    ray.remote(max_concurrency=int(os.environ.get("RAY_MAX_CONCURRENCY", 1000)))(RawBinarySearchEnvironment),
)
BinarySearchEnvironmentProxy = ActorProxy[RawBinarySearchEnvironment]
