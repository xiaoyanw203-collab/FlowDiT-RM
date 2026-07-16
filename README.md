# FlowDiT-RM

Official implementation of the manuscript:

**FlowDiT-RM: A Propagation-Prior Flow Transformer for Zero-Sampling 3D Radio Map Generation**

Submitted to *IEEE Transactions on Cognitive Communications and Networking*.

## Repository Structure

```text
FlowDiT-RM/
├── docx/
├── scripts/
├── flowdit_rm/
│   ├── datasets/
│   ├── models/
│   ├── physics/
│   ├── training/
│   ├── evaluation/
├── requirements.txt
└── README.md

Baselines and other ablation models were intentionally excluded.
```

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
