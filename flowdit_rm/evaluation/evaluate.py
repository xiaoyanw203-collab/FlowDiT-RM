import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


from flowdit_rm.datasets.dataset import RadioMapDataset
from flowdit_rm.models.ablation_cond_dit import AblationCondDiT
from flowdit_rm.physics.phys_dit import RectifiedFlowWrapper

HEIGHT_MAX_DROP = 1439.29
EVAL_CFG_SCALE = 1.0

EVAL_BATCH_SIZE = 1

# Number of GPU warm-up iterations; excluded from latency
WARMUP_ITERS = 20

EPS = 1e-12


def test_cfg_visual(model, rf_pipeline, test_loader, device, save_dir="."):
    """
    Generate the CFG visualization comparison figure.
    This part is excluded from latency.
    """
    os.makedirs(save_dir, exist_ok=True)

    x_1, x_0, cond, freq = next(iter(test_loader))
    x_1 = x_1.to(device, non_blocking=True)
    x_0 = x_0.to(device, non_blocking=True)
    cond = cond.to(device, non_blocking=True)
    freq = freq.to(device, non_blocking=True)

    x_1_single = x_1[0:1]
    x_0_single = x_0[0:1]
    cond_single = cond[0:1]
    freq_single = freq[0:1]

    with torch.no_grad():
        pred_cfg_1 = rf_pipeline.sample_euler(
            x_0_single,
            cond_single,
            freq_single,
            num_steps=1,
            cfg_scale=1.0
        )

        pred_cfg_enhanced = rf_pipeline.sample_euler(
            x_0_single,
            cond_single,
            freq_single,
            num_steps=1,
            cfg_scale=EVAL_CFG_SCALE
        )

    img_fspl = np.clip((x_0_single[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
    img_cfg1 = np.clip((pred_cfg_1[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
    img_cfgenhanced = np.clip((pred_cfg_enhanced[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)
    img_gt = np.clip((x_1_single[0, 0].cpu().numpy() + 1.0) / 2.0, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(img_fspl, cmap="jet", vmin=0, vmax=1)
    axes[0].set_title("1. Input: FSPL Prior")
    axes[0].axis("off")

    im1 = axes[1].imshow(img_cfg1, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("2. Concat-Cond (CFG = 1.0)")
    axes[1].axis("off")

    im2 = axes[2].imshow(img_cfgenhanced, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title(f"3. Concat-Cond (CFG = {EVAL_CFG_SCALE})")
    axes[2].axis("off")

    im3 = axes[3].imshow(img_gt, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("4. Ground Truth")
    axes[3].axis("off")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im3, cax=cbar_ax)

    save_path = os.path.join(save_dir, "AblationCond_CFG_Comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ Comparison figure saved to: {save_path}")


def warmup_model(rf_pipeline, test_loader, device):
    """
    GPU warm-up, excluded from latency.
    """
    print(f"🔥 GPU warm-up: {WARMUP_ITERS} iterations...")

    with torch.no_grad():
        for k, (x_1, x_0, cond, freq) in enumerate(test_loader):
            if k >= WARMUP_ITERS:
                break

            x_0 = x_0.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)

            _ = rf_pipeline.sample_euler(
                x_0,
                cond,
                freq,
                num_steps=1,
                cfg_scale=EVAL_CFG_SCALE
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("✅ Warm-up finished.")


def timed_sample_euler(rf_pipeline, x_0, cond, freq, device):
    """
    Use CUDA events to measure pure model inference time for sample_euler.
    Returns:
        generated_minus1_1: model output
        batch_time_ms: inference time for the current batch, in ms
    """
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        starter.record()

        generated_minus1_1 = rf_pipeline.sample_euler(
            x_0,
            cond,
            freq,
            num_steps=1,
            cfg_scale=EVAL_CFG_SCALE
        )

        ender.record()
        torch.cuda.synchronize()

        batch_time_ms = starter.elapsed_time(ender)

    else:
        start_time = time.perf_counter()

        generated_minus1_1 = rf_pipeline.sample_euler(
            x_0,
            cond,
            freq,
            num_steps=1,
            cfg_scale=EVAL_CFG_SCALE
        )

        batch_time_ms = (time.perf_counter() - start_time) * 1000.0

    return generated_minus1_1, batch_time_ms


def evaluate():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting evaluation. Device: {device}")
    print(f"🔥 Current CFG scale for evaluation = {EVAL_CFG_SCALE}")

    test_dataset = RadioMapDataset(
        data_dir="/workspace/MapLevel_Split_3Way/test",
        terrain_max_drop=HEIGHT_MAX_DROP
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = AblationCondDiT(input_size=128, depth=28).to(device)
    rf_pipeline = RectifiedFlowWrapper(model)

    ckpt_path = "/workspace/src/weights/checkpoints/best.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found文件：{ckpt_path}。Please finish running train.py first！"
        )

    checkpoint = torch.load(ckpt_path, map_location=device)

    if "ema_shadow" in checkpoint and checkpoint["ema_shadow"]:
        missing, unexpected = model.load_state_dict(checkpoint["ema_shadow"], strict=False)
        print("✅ Loaded EMA weights。")
    elif "model_state_dict" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("✅ EMA weights not found; loaded model_state_dict。")
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded weights as a regular state_dict。")

    if missing:
        print(f"⚠️ Keys missing from checkpoint: {missing}")
    if unexpected:
        print(f"⚠️ Unexpected keys in checkpoint: {unexpected}")

    model.eval()

    # Visualization is excluded from latency
    test_cfg_visual(model, rf_pipeline, test_loader, device, save_dir=".")

    # Warm-up is excluded from latency
    warmup_model(rf_pipeline, test_loader, device)

    total_nmse = 0.0
    total_nmae = 0.0
    total_ssim = 0.0
    total_psnr = 0.0

    # Optional: also compute RMSE for inspection
    total_se_for_rmse = 0.0
    total_pixels_for_rmse = 0

    total_inference_time_ms = 0.0
    num_samples = 0
    num_channels = 0

    with torch.no_grad():
        for x_1, x_0, cond, freq in tqdm(test_loader, desc=f"Running Concat-Cond inference (CFG={EVAL_CFG_SCALE})"):
            x_1 = x_1.to(device, non_blocking=True)
            x_0 = x_0.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            freq = freq.to(device, non_blocking=True)

            generated_minus1_1, batch_time_ms = timed_sample_euler(
                rf_pipeline,
                x_0,
                cond,
                freq,
                device
            )

            bs = x_1.shape[0]
            total_inference_time_ms += batch_time_ms
            num_samples += bs

            # Post-processing and metric computation are excluded from latency
            pred_0_1 = torch.clamp((generated_minus1_1 + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()
            target_0_1 = torch.clamp((x_1 + 1.0) / 2.0, 0.0, 1.0).cpu().numpy()

            bs, c, h, w = pred_0_1.shape
            num_channels = c

            for i in range(bs):
                pred_i = pred_0_1[i]
                target_i = target_0_1[i]

                # ==================================================
                # Correct sample-wise NMSE / NMAE
                # NMSE = ||pred - target||_2^2 / ||target||_2^2
                # NMAE = ||pred - target||_1 / ||target||_1
                # ==================================================
                se = np.sum((pred_i - target_i) ** 2)
                gt_energy = np.sum(target_i ** 2)

                ae = np.sum(np.abs(pred_i - target_i))
                gt_l1 = np.sum(np.abs(target_i))

                sample_nmse = se / (gt_energy + EPS)
                sample_nmae = ae / (gt_l1 + EPS)

                total_nmse += sample_nmse
                total_nmae += sample_nmae

                # Optional RMSE statistics
                total_se_for_rmse += se
                total_pixels_for_rmse += pred_i.size

                # SSIM / PSNR: compute per height layer and then average
                sample_ssim = 0.0
                sample_psnr = 0.0

                for j in range(c):
                    p_img = pred_i[j]
                    t_img = target_i[j]

                    sample_ssim += ssim(t_img, p_img, data_range=1.0)
                    sample_psnr += psnr(t_img, p_img, data_range=1.0)

                total_ssim += sample_ssim / c
                total_psnr += sample_psnr / c

    avg_nmse = total_nmse / num_samples
    avg_nmae = total_nmae / num_samples
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    avg_rmse = np.sqrt(total_se_for_rmse / (total_pixels_for_rmse + EPS))

    avg_time_per_sample = total_inference_time_ms / num_samples

    print("\n" + "=" * 60)
    print("🏆 Evaluation results")
    print("=" * 60)
    print(f"🔥 CFG scale       : {EVAL_CFG_SCALE}")
    print(f"📊 Number of spatial channels/高度层  : {num_channels}")
    print(f"🌍 Average NMSE          : {avg_nmse:.5f}")
    print(f"🎯 Average NMAE          : {avg_nmae:.5f}")
    print(f"📌 Average RMSE (optional)    : {avg_rmse:.5f}")
    print(f"👁️  Average SSIM (3D-Avg): {avg_ssim:.4f}")
    print(f"🔊 Average PSNR (3D-Avg) : {avg_psnr:.2f} dB")
    print("-" * 60)
    print(f"⚡ Inference latency   : {avg_time_per_sample:.2f} ms / sample")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
