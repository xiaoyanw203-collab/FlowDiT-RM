import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
# Enable TF32 acceleration 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from flowdit_rm.datasets.dataset import RadioMapDataset
from flowdit_rm.models.ablation_cond_dit import AblationCondDiT
from flowdit_rm.physics.phys_dit import RectifiedFlowWrapper

# ==========================================================
# EMA (exponential moving average) weight manager
# ==========================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            self.backup[name] = param.data
            param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            param.data = self.backup[name]
        self.backup = {}

# ==========================================================
# Helper: visualize inference results
# ==========================================================
def save_visualization(x_0, x_1, generated, epoch, save_dir="ablation_cond_training_vis"):
    os.makedirs(save_dir, exist_ok=True)
    
    fspl_img = ((x_0[0, 0].cpu().numpy() + 1.0) / 2.0)
    gt_img = ((x_1[0, 0].cpu().numpy() + 1.0) / 2.0)
    pred_img = ((generated[0, 0].cpu().numpy() + 1.0) / 2.0)
    
    fspl_img = np.clip(fspl_img, 0, 1)
    gt_img = np.clip(gt_img, 0, 1)
    pred_img = np.clip(pred_img, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(fspl_img, cmap='jet', vmin=0, vmax=1)
    axes[0].set_title("Input: FSPL Prior")
    
    im2 = axes[1].imshow(pred_img, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"Concat-Cond Output (Epoch {epoch})") # Title update
    
    im3 = axes[2].imshow(gt_img, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Ground Truth (Target)")

    plt.colorbar(im3, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()

def _training_state_dict(epoch, model, ema, optimizer, scheduler, best_val_loss, epochs_no_improve):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": {k: v.detach().cpu() for k, v in ema.shadow.items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "epochs_no_improve": epochs_no_improve,
        "val_loss": best_val_loss,
    }

def _load_checkpoint(path, device, model, ema, optimizer, scheduler):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if "ema_shadow" in ckpt and ckpt["ema_shadow"]:
        for name, tensor in ckpt["ema_shadow"].items():
            ema.shadow[name] = tensor.to(device)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
    epochs_no_improve = ckpt.get("epochs_no_improve", 0)
    start_epoch = int(ckpt["epoch"]) + 1
    return start_epoch, best_val_loss, epochs_no_improve

def train(args):
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    TERRAIN_MAX_DROP = args.terrain_max_drop
    SAVE_DIR = args.save_dir
    CKPT_BEST = args.ckpt_best
    CKPT_LAST = args.ckpt_last
    os.makedirs(SAVE_DIR, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(3)
        device = torch.device("cuda:3")
    else:
        device = torch.device("cpu")
    phys = args.physical_gpu or os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"🚀 Starting training. PyTorch device: {device} | Selected physical GPU: {phys}")

    train_dataset = RadioMapDataset(data_dir=args.train_data, terrain_max_drop=TERRAIN_MAX_DROP)
    val_dataset = RadioMapDataset(data_dir=args.val_data, terrain_max_drop=TERRAIN_MAX_DROP)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    fixed_val_batch = next(iter(val_loader))

    # Instantiate FlowDiT
    model = AblationCondDiT(input_size=args.input_size, depth=args.depth).to(device)
    rf_pipeline = RectifiedFlowWrapper(model)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    ema = EMA(model, decay=args.ema_decay)

    best_val_loss = float('inf')
    early_stop_patience = args.early_stop_patience
    epochs_no_improve = 0
    start_epoch = 1

    if args.resume is not None:
        resume_path = os.path.join(SAVE_DIR, CKPT_LAST) if args.resume == "last" else args.resume
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        start_epoch, best_val_loss, epochs_no_improve = _load_checkpoint(
            resume_path, device, model, ema, optimizer, scheduler
        )
        print(f"📂 Resumed from checkpoint: {resume_path} | next epoch={start_epoch} | best_val_loss={best_val_loss:.5f}")

    metrics_path = os.path.join(SAVE_DIR, "ablation_cond_metrics.txt") # Separate log file
    if start_epoch == 1:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"# run_start: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"# device: {device} | MODE: TF32 Pure Float32 | MODEL: Concat-Cond Ablation\n")
            f.write(f"epoch\ttrain_loss\tval_loss\tlr\tbest_val_loss\tepochs_no_improve\ttime\n")
    else:
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"# resume_at: {datetime.now().isoformat(timespec='seconds')} from epoch {start_epoch}\n")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for x_1, x_0, cond, freq in pbar:
            x_1, x_0, cond, freq = x_1.to(device), x_0.to(device), cond.to(device), freq.to(device)
            
            optimizer.zero_grad()
            loss = rf_pipeline.get_train_loss(x_1, x_0, cond, freq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                ema.update(model)
            
            train_loss_total += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})
            
        avg_train_loss = train_loss_total / len(train_loader)
        scheduler.step()

        model.eval()
        ema.apply_shadow(model)
        val_loss_total = 0.0
        
        with torch.no_grad():
            for x_1, x_0, cond, freq in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]  ", leave=False):
                x_1, x_0, cond, freq = x_1.to(device), x_0.to(device), cond.to(device), freq.to(device)
                loss = rf_pipeline.get_train_loss(x_1, x_0, cond, freq)
                val_loss_total += loss.item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"📈 Concat-Cond Epoch {epoch} summary: Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch % 5 == 0 or epoch == 1:
            x_1_fix, x_0_fix, cond_fix, freq_fix = [tensor.to(device) for tensor in fixed_val_batch]
            with torch.no_grad():
                generated = rf_pipeline.sample_euler(x_0_fix, cond_fix, freq_fix, num_steps=10)
            save_visualization(x_0_fix, x_1_fix, generated, epoch, save_dir=args.vis_dir)

        ema.restore(model)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(_training_state_dict(epoch, model, ema, optimizer, scheduler, best_val_loss, epochs_no_improve),
                os.path.join(SAVE_DIR, CKPT_BEST),
            )
        else:
            epochs_no_improve += 1

        torch.save(_training_state_dict(epoch, model, ema, optimizer, scheduler, best_val_loss, epochs_no_improve),
            os.path.join(SAVE_DIR, CKPT_LAST),
        )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{avg_train_loss:.6f}\t{avg_val_loss:.6f}\t{scheduler.get_last_lr()[0]:.8e}\t{best_val_loss:.6f}\t{epochs_no_improve}\t{datetime.now().isoformat(timespec='seconds')}\n")

        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Validation loss did not improve for {early_stop_patience} epochs; early stopping triggered！")
            break

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Model training")
    p.add_argument("--physical-gpu", type=str, default=None)
    p.add_argument("--resume", nargs="?", const="last", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--terrain-max-drop", type=float, default=1439.29)
    # Separate save directory
    p.add_argument("--save-dir", type=str, default="/workspace/src/weights/checkpoints")
    p.add_argument("--ckpt-best", type=str, default="best.pth")
    p.add_argument("--ckpt-last", type=str, default="last.pth")
    p.add_argument("--train-data", type=str, default="/workspace/MapLevel_Split_3Way/train")
    p.add_argument("--val-data", type=str, default="/workspace/MapLevel_Split_3Way/val")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--depth", type=int, default=28)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    # Separate visualization directory
    p.add_argument("--vis-dir", type=str, default="training_vis")
    return p

if __name__ == "__main__":
    _cli = _build_arg_parser().parse_args()
    train(_cli)
