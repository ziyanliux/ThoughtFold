<h1 align="center">ThoughtFold<br><small>Efficient CoT Optimization via Binary Search and Attention-Guided Pruning</small></h1>

<p align="center"><em>Online DPO Training for Chain-of-Thought Length Reduction in Reasoning Models</em></p>

<p align="center"><b>Accepted by ICML 2026</b></p>

---

## 🚀 Motivation

Long Chain-of-Thought (CoT) reasoning improves accuracy but incurs substantial inference cost. Naively truncating or distilling CoT often degrades correctness.

> **ThoughtFold** finds the shortest CoT that preserves correctness — *online, during RL training* — and constructs masked DPO pairs to teach the model to reason more concisely without losing accuracy.

---

## 💡 Key Idea

ThoughtFold performs **two-phase CoT pruning** within the GRPO training loop:

- **Phase 1 — Tail Truncation (Binary Search):** For each correct sample, binary search on CoT length to find the shortest prefix that still produces correct answers above a confidence threshold.
- **Phase 2 — Internal Folding (Attention-Guided):** Use attention scores to identify and remove low-importance reasoning sentences, then binary search on the retention ratio.

Each pruning iteration produces a **Masked DPO pair**:
- ✅ **Case 1 (Pruned & Correct):** shorter response = chosen, longer response = rejected. Loss is applied only to the pruned region.
- ❌ **Case 2 (Overjump):** over-pruned incorrect response = rejected, last correct response = chosen. Loss targets the answer portion.

---

## 🧩 Method

<p align="center"><img src="../figs/method_overview.png" width="90%"></p>

1. **Generate & Judge** — Standard GRPO rollout with reward verification.
2. **Binary Search (Phase 1)** — For correct samples, iteratively halve CoT length, validate with repeated sampling + judging, and track the shortest correct prefix.
3. **Attention Pruning (Phase 2)** — Obtain per-token attention scores for the shortest response, aggregate to sentence level, and binary search on top-k retention ratio.
4. **DPO Training** — Constructed masked DPO pairs are shuffled with GRPO samples and trained jointly with a hybrid loss.

---

## 📦 Project Structure

```
thoughtfold/
├── __init__.py
├── main.py                          # Standard GRPO training entry
├── thoughtfold_train.py             # ThoughtFold entry (DPO + Binary Search)
└── binsearch/
    ├── __init__.py
    ├── binary_search_environment.py # Core algorithm: two-phase pruning
    ├── binary_search_trainer.py     # DPO Trainer with masked label construction
    └── utils.py
```

---

## ⚙️ Usage

### Configuration

Key binary search parameters in config file:

```python
enable_binary_search = True
binary_search_config = {
    'repeat': 4,                        # Validation sampling repeat
    'threshold': 0.7,                   # Correctness threshold for acceptance
    'max_iterations': 5,                # Max binary search iterations (Phase 1)
    'min_cot_length': 300,              # Minimum CoT length to attempt pruning
    'enable_fine_grained_pruning': True, # Enable Phase 2
    'topk_search_min': 0.1,            # Min retention ratio (Phase 2)
    'topk_search_max': 0.9,            # Max retention ratio (Phase 2)
    'topk_search_iterations': 5,        # Max iterations (Phase 2)
    'pruning_repeat': 4,               # Validation repeat (Phase 2)
}
```

### Run Training

```bash
python -m thoughtfold.thoughtfold_train <config.py> \
    --ray-cluster-url ray://<master>:10001 \
    --work-dir ./work_dir \
    --num-workers 8
```

---

## 📌 Citation

```bibtex
@inproceedings{thoughtfold2026,
    title={ThoughtFold: Efficient Chain-of-Thought Optimization via Binary Search and Attention-Guided Pruning},
    author={},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2026}
}
```
