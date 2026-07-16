# FlowDiT-RM

This repository contains the cleaned main-experiment code from `DiT_Flow`.

Included code:

- Main model: `flowdit_rm/models/ablation_cond_dit.py`
- Rectified-flow wrapper: `flowdit_rm/physics/phys_dit.py`
- Dataset loader: `flowdit_rm/datasets/dataset.py`
- Training entry point: `flowdit_rm/training/train_ablation_cond.py`
- Evaluation entry point: `flowdit_rm/evaluation/cond.py`

Baselines and other ablation models were intentionally excluded.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
bash scripts/train_ablation_cond.sh --help
```

## Evaluate

```bash
bash scripts/eval_cond.sh --help
```
