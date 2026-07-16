import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src" / "data"))
sys.path.insert(0, str(_root / "src" / "model"))

from flowdit_rm.datasets.dataset import RadioMapDataset
from flowdit_rm.models.ablation_cond_dit import AblationCondDiT
from flowdit_rm.physics.phys_dit import RectifiedFlowWrapper

# =========================
# Basic configuration
# =========================

HEIGHT_MAX_DROP = 1439.29

TEST_DATA_DIR = "/workspace/MapLevel_Split_3Way/test"
CKPT_PATH = "/workspace/src/weights/checkpoints/best.pth"

BATCH_SIZE = 2
NUM_STEPS = 1

# CFG sweep
CFG_LIST = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]

PERCENTILE_THRESHOLDS = [70, 80, 90]


# =========================
# Coverage metric computation
# =========================


def compute_coverage_metrics(pred_map, gt_map, threshold):
    """
    pred_map, gt_map: [H, W], already in [0, 1]
    threshold: derived from the GT percentile

    Assume larger values indicate stronger signal: coverage = map > threshold
    """
    pred_cov = pred_map > threshold
    gt_cov = gt_map > threshold

    tp = np.logical_and(pred_cov, gt_cov).sum()
    fp = np.logical_and(pred_cov, np.logical_not(gt_cov)).sum()
    fn = np.logical_and(np.logical_not(pred_cov), gt_cov).sum()
    tn = np.logical_and(np.logical_not(pred_cov), np.logical_not(gt_cov)).sum()

    eps = 1e-8

    iou = tp / (tp + fp + fn + eps)
    fcr = fp / (tp + fp + eps)
    mcr = fn / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": float(iou),
        "fcr": float(fcr),
        "mcr": float(mcr),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "acc": float(acc),
    }


def init_metric_dict(percentiles):
    metric_sum = {}
    for q in percentiles:
        key = f"Q{q}"
        metric_sum[key] = {
            "iou": 0.0,
            "fcr": 0.0,
            "mcr": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "acc": 0.0,
            "count": 0,
        }
    return metric_sum


def update_metric_sum(metric_sum, key, metrics):
    for k in ["iou", "fcr", "mcr", "f1", "precision", "recall", "acc"]:
        metric_sum[key][k] += metrics[k]
    metric_sum[key]["count"] += 1


def summarize_metric_dict(metric_sum):
    summary = {}
    avg_metrics = {
        "iou": 0.0,
        "fcr": 0.0,
        "mcr": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "acc": 0.0,
    }
    keys = list(metric_sum.keys())

    for key, values in metric_sum.items():
        count = max(values["count"], 1)
        summary[key] = {}
        for m in ["iou", "fcr", "mcr", "f1", "precision", "recall", "acc"]:
            summary[key][m] = values[m] / count
            avg_metrics[m] += summary[key][m]

    n = max(len(keys), 1)
    summary["Avg"] = {m: avg_metrics[m] / n for m in avg_metrics}
    return summary


def print_single_cfg_table(cfg_scale, summary):
    print("\n" + "=" * 100)
    print(f"🏆 Concat-Cond Ablation | GT-Percentile Coverage | CFG={cfg_scale}")
    print("=" * 100)
    print(
        f"{'Threshold':>12} | "
        f"{'IoU↑':>10} | "
        f"{'FCR↓':>10} | "
        f"{'MCR↓':>10} | "
        f"{'F1↑':>10} | "
        f"{'Prec↑':>10} | "
        f"{'Recall↑':>10} | "
        f"{'Acc↑':>10}"
    )
    print("-" * 100)

    for key in list(summary.keys()):
        values = summary[key]
        print(
            f"{key:>12} | "
            f"{values['iou']:10.4f} | "
            f"{values['fcr']:10.4f} | "
            f"{values['mcr']:10.4f} | "
            f"{values['f1']:10.4f} | "
            f"{values['precision']:10.4f} | "
            f"{values['recall']:10.4f} | "
            f"{values['acc']:10.4f}"
        )

    print("=" * 100)


def print_cfg_sweep_summary(all_results):
    print("\n" + "=" * 110)
    print("📌 Concat-Cond | CFG Sweep Summary | Main threshold = Q80")
    print("=" * 110)
    print(
        f"{'CFG':>8} | "
        f"{'IoU@Q80↑':>10} | "
        f"{'FCR@Q80↓':>10} | "
        f"{'MCR@Q80↓':>10} | "
        f"{'F1@Q80↑':>10} | "
        f"{'Avg IoU↑':>10} | "
        f"{'Avg FCR↓':>10} | "
        f"{'Avg MCR↓':>10} | "
        f"{'Avg F1↑':>10}"
    )
    print("-" * 110)

    for cfg, summary in all_results.items():
        q80 = summary["Q80"]
        avg = summary["Avg"]
        print(
            f"{cfg:8.2f} | "
            f"{q80['iou']:10.4f} | "
            f"{q80['fcr']:10.4f} | "
            f"{q80['mcr']:10.4f} | "
            f"{q80['f1']:10.4f} | "
            f"{avg['iou']:10.4f} | "
            f"{avg['fcr']:10.4f} | "
            f"{avg['mcr']:10.4f} | "
            f"{avg['f1']:10.4f}"
        )

    print("=" * 110)


def evaluate_coverage_one_cfg(model, rf_pipeline, test_loader, device, cfg_scale):
    metric_sum = init_metric_dict(PERCENTILE_THRESHOLDS)
    total_inference_time = 0.0
    num_samples = 0
    num_channels = 0
    model.eval()

    with torch.no_grad():
        for x_1, x_0, cond, freq in tqdm(
            test_loader,
            desc=f"Ablation-Cond Coverage CFG={cfg_scale}",
        ):
            x_1 = x_1.to(device)
            x_0 = x_0.to(device)
            cond = cond.to(device)
            freq = freq.to(device)

            start_time = time.perf_counter()
            generated_minus1_1 = rf_pipeline.sample_euler(
                x_0,
                cond,
                freq,
                num_steps=NUM_STEPS,
                cfg_scale=cfg_scale,
            )
            batch_time = time.perf_counter() - start_time
            total_inference_time += batch_time

            pred_0_1 = torch.clamp(
                (generated_minus1_1 + 1.0) / 2.0, 0.0, 1.0
            ).cpu().numpy()
            target_0_1 = torch.clamp((x_1 + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()

            bs, c, h, w = pred_0_1.shape
            num_channels = c

            for i in range(bs):
                for j in range(c):
                    pred_map = pred_0_1[i, j]
                    gt_map = target_0_1[i, j]
                    for q in PERCENTILE_THRESHOLDS:
                        threshold = np.percentile(gt_map, q)
                        key = f"Q{q}"
                        metrics = compute_coverage_metrics(
                            pred_map=pred_map,
                            gt_map=gt_map,
                            threshold=threshold,
                        )
                        update_metric_sum(metric_sum, key, metrics)
                num_samples += 1

    avg_time_per_sample = (total_inference_time / max(num_samples, 1)) * 1000
    summary = summarize_metric_dict(metric_sum)
    return summary, avg_time_per_sample, num_samples, num_channels


def evaluate_coverage_cfg_sweep():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    if not Path(CKPT_PATH).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}。Please run train_ablation_cond.py first。"
        )

    test_dataset = RadioMapDataset(
        data_dir=TEST_DATA_DIR,
        terrain_max_drop=HEIGHT_MAX_DROP,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = AblationCondDiT(input_size=128, depth=28).to(device)
    rf_pipeline = RectifiedFlowWrapper(model)

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["ema_shadow"], strict=False)
    if missing:
        print(f"⚠️ Keys missing from checkpoint: {missing}")
    if unexpected:
        print(f"⚠️ Unused keys in checkpoint: {unexpected}")

    model.eval()
    all_results = {}

    for cfg_scale in CFG_LIST:
        summary, avg_time, num_samples, num_channels = evaluate_coverage_one_cfg(
            model=model,
            rf_pipeline=rf_pipeline,
            test_loader=test_loader,
            device=device,
            cfg_scale=cfg_scale,
        )
        all_results[cfg_scale] = summary
        print_single_cfg_table(cfg_scale, summary)
        print(f"\n📊 CFG={cfg_scale} Number of samples: {num_samples}")
        print(f"📊 Number of spatial channels: {num_channels}")
        print(f"⚡ Average inference time: {avg_time:.2f} ms / sample")

    print_cfg_sweep_summary(all_results)


if __name__ == "__main__":
    evaluate_coverage_cfg_sweep()
