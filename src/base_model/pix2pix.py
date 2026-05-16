import os
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.amp import GradScaler, autocast

import piq
from tqdm import tqdm
from math import log10
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================
CHECKPOINT_DIR = "./checkpoints"
LOAD_DIR = "./checkpoints"
DATASET_ROOT = "archive"  # Path to KAIST dataset
EPOCHS = 21               # Constraint: Keep 20
MAX_SAMPLES = 20000       # Constraint: Keep 20k
FINE_SIZE = 256           # Changed: 256 for better stability/batch size
BATCH_SIZE = 6            # Changed: Increased from 2 -> 6 (fits in ~4GB VRAM at 256px)
NUM_WORKERS = 2
LR = 2e-4
LAMBDA_L1 = 100
LAMBDA_SSIM = 5.0         # Changed: Increased from 1.0 to prioritize structure


def evaluate_and_show(epoch, G, val_set, device, num_extra_examples=5):
    G.eval()

    # =============================
    # Evaluate on first validation sample
    # =============================
    if len(val_set) > 0:
        example = val_set[0]
        with torch.no_grad():
            real_A = example["A"].unsqueeze(0).to(device)
            real_B = example["B"].unsqueeze(0).to(device)
            fake_B = G(real_A)

            # ----- Metrics -----
            # Ensure values are in [0, 1] for metric calculation
            fake_B_01 = to_01(fake_B)
            real_B_01 = to_01(real_B)
            
            ssim_val = ssim(fake_B_01, real_B_01).item()
            
            # Avoid divide by zero in PSNR
            mse = nn.MSELoss()(fake_B, real_B).item()
            psnr_val = 10 * log10(1 / (mse + 1e-8))
            
            lpips_val = piq.LPIPS()(fake_B, real_B).item()

            print(f"\n===== Epoch {epoch} Metrics =====")
            print(f"SSIM : {ssim_val:.4f}")
            print(f"PSNR : {psnr_val:.4f}")
            print(f"LPIPS: {lpips_val:.4f}")

            # ----- Visualization -----
            fa = to_01(real_A[0]).cpu().permute(1,2,0)
            fb = to_01(fake_B[0]).cpu().permute(1,2,0)
            rb = to_01(real_B[0]).cpu().permute(1,2,0)

            plt.figure(figsize=(15,5))
            plt.subplot(1,3,1); plt.title("RGB"); plt.imshow(fa); plt.axis("off")
            plt.subplot(1,3,2); plt.title("Generated IR"); plt.imshow(fb, cmap="gray"); plt.axis("off")
            plt.subplot(1,3,3); plt.title("Real IR"); plt.imshow(rb, cmap="gray"); plt.axis("off")
            plt.savefig(f"results/epoch_{epoch}_example.png")
            plt.close()

    # =============================
    # SHOW EXTRA SAMPLES
    # =============================
    print(f"\nShowing {num_extra_examples} extra generated examples…")
    generate_examples(G, val_set, device, n=num_extra_examples, epoch=epoch)

# Convert [-1,1] → [0,1]
def to_01(x):
    return (x + 1) / 2

def generate_examples(G, dataset, device, n=5, epoch=0):
    """
    Shows n random samples from dataset using the currently loaded model G.
    """
    G.eval()
    
    n = min(n, len(dataset))
    if n == 0: return

    indices = random.sample(range(len(dataset)), n)

    for idx in indices:
        batch = dataset[idx]
        real_A = batch["A"].unsqueeze(0).to(device)
        real_B = batch["B"].unsqueeze(0).to(device)

        with torch.no_grad():
            fake_B = G(real_A)

        fa = to_01(real_A[0]).cpu().permute(1,2,0)
        fb = to_01(fake_B[0]).cpu().permute(1,2,0)
        rb = to_01(real_B[0]).cpu().permute(1,2,0)

        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1); plt.title("RGB Input"); plt.imshow(fa); plt.axis("off")
        plt.subplot(1, 3, 2); plt.title("Generated IR"); plt.imshow(fb, cmap="gray"); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Real IR"); plt.imshow(rb, cmap="gray"); plt.axis("off")

        plt.savefig(f"results/epoch_{epoch}_example_{idx}.png")
        plt.close()

def get_latest_checkpoint(folder=LOAD_DIR):
    if not os.path.exists(folder):
        return None
    files = glob.glob(os.path.join(folder, "checkpoint_epoch*.pth"))
    if not files:
        return None
    files.sort(key=lambda x: int(os.path.basename(x).split("epoch")[1].split(".")[0]))
    return files[-1]

class KAISTKaggleDataset(Dataset):
    
    def __init__(self, root=DATASET_ROOT, fine_size=FINE_SIZE, max_samples=MAX_SAMPLES):
        self.root = root
        self.fine_size = fine_size
        self.samples = []

        print("Scanning dataset directories...")
        for set_name in sorted(os.listdir(root)):
            set_path = os.path.join(root, set_name)
            if not os.path.isdir(set_path): continue

            for seq_name in sorted(os.listdir(set_path)):
                seq_path = os.path.join(set_path, seq_name)
                visible_dir = os.path.join(seq_path, "visible")
                lwir_dir = os.path.join(seq_path, "lwir")

                if not os.path.isdir(visible_dir) or not os.path.isdir(lwir_dir): continue

                visible_files = sorted(os.listdir(visible_dir))
                for fname in visible_files:
                    vis = os.path.join(visible_dir, fname)
                    ir = os.path.join(lwir_dir, fname)
                    if os.path.isfile(vis) and os.path.isfile(ir):
                        self.samples.append((vis, ir))

        total_found = len(self.samples)
        print(f"Total pairs found: {total_found}")

        if total_found > max_samples:
            print(f"Limiting dataset to {max_samples} random samples...")
            random.seed(42) 
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
        
        print(f"Final Dataset Size: {len(self.samples)} image pairs.")

        # Transforms applied individually after joint transforms
        self.norm_rgb = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.norm_ir = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, ir_path = self.samples[idx]

        rgb = Image.open(rgb_path).convert("RGB")
        ir = Image.open(ir_path)
        ir = ImageOps.grayscale(ir)

        # ======================================
        # CHANGED: Joint Random Augmentation
        # ======================================
        
        # 1. Random Crop (Ensure same parameters for both)
        # We need images > fine_size. Most KAIST are 640x512, so this is safe.
        if rgb.size[0] > self.fine_size and rgb.size[1] > self.fine_size:
            i, j, h, w = T.RandomCrop.get_params(rgb, output_size=(self.fine_size, self.fine_size))
            rgb = TF.crop(rgb, i, j, h, w)
            ir = TF.crop(ir, i, j, h, w)
        else:
            # Fallback resize if smaller (unlikely for this dataset)
            rgb = TF.resize(rgb, (self.fine_size, self.fine_size))
            ir = TF.resize(ir, (self.fine_size, self.fine_size))

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            ir = TF.hflip(ir)

        return {
            "A": self.norm_rgb(rgb),
            "B": self.norm_ir(ir),
            "A_path": rgb_path,
            "B_path": ir_path
        }

class UNetBlock(nn.Module):
    
    def __init__(self, in_c, out_c, submodule=None, outermost=False, innermost=False):
        super().__init__()
        down = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, True)
        )
        if innermost:
            up = nn.Sequential(
                nn.ConvTranspose2d(out_c, in_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(True)
            )
            self.model = nn.Sequential(down, up)
        elif outermost:
            self.model = nn.Sequential(
                down,
                submodule,
                nn.ConvTranspose2d(out_c, 1, 4, 2, 1),
                nn.Tanh()
            )
        else:
            up = nn.Sequential(
                nn.ConvTranspose2d(out_c, in_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(True)
            )
            self.model = nn.Sequential(down, submodule, up)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


def build_unet(input_nc=3):
    innermost = UNetBlock(512, 512, None, innermost=True)
    block = UNetBlock(256, 512, innermost)
    block = UNetBlock(128, 256, block)
    block = UNetBlock(64, 128, block)
    model = UNetBlock(input_nc, 64, block, outermost=True)
    return model


class PatchDiscriminator(nn.Module):
    
    def __init__(self, in_nc=4, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    G = build_unet().to(device)
    D = PatchDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    scaler_G = GradScaler('cuda')
    scaler_D = GradScaler('cuda')

    # ===== Dataset =====
    dataset = KAISTKaggleDataset(fine_size=FINE_SIZE, max_samples=MAX_SAMPLES)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE,  # Updated
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=1, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # ===== Scheduler =====
    # CHANGED: Linear Decay Scheduler
    # Keep LR constant for first 10 epochs, then decay linearly to 0
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - 10) / float(10 + 1)
        return lr_l
    
    scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
    scheduler_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_rule)

    # ===== Load Checkpoint =====
    start_epoch = 0
    latest_ckpt = get_latest_checkpoint()

    if latest_ckpt:
        print(f"Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        
        # Load scheduler state if possible, else reset
        if "sched_G" in ckpt:
            scheduler_G.load_state_dict(ckpt["sched_G"])
            scheduler_D.load_state_dict(ckpt["sched_D"])
    else:
        print("Starting from scratch.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    bce = nn.BCEWithLogitsLoss()

    # ===== Training Loop =====
    for epoch in range(start_epoch, EPOCHS):
        G.train()
        D.train()

        pbar = tqdm(train_loader)
        for batch in pbar:
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # ========== DISCRIMINATOR ==========
            opt_D.zero_grad()
            with autocast('cuda'):
                fake_B = G(real_A)
                pred_real = D(torch.cat([real_A, real_B], 1))
                pred_fake = D(torch.cat([real_A, fake_B.detach()], 1))

                # CHANGED: Label Smoothing (Real = 0.9 instead of 1.0)
                loss_D = 0.5 * (
                    bce(pred_real, torch.ones_like(pred_real) * 0.9) +
                    bce(pred_fake, torch.zeros_like(pred_fake))
                )
            
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # ========== GENERATOR ==========
            opt_G.zero_grad()
            with autocast('cuda'):
                pred_fake = D(torch.cat([real_A, fake_B], 1))
                
                loss_G_GAN = bce(pred_fake, torch.ones_like(pred_fake))
                loss_L1 = nn.L1Loss()(fake_B, real_B)
                loss_S = 1 - ssim(to_01(fake_B), to_01(real_B))

                # CHANGED: Increased SSIM influence
                loss_G = loss_G_GAN + LAMBDA_L1 * loss_L1 + LAMBDA_SSIM * loss_S
            
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            pbar.set_description(f"E{epoch} | D={loss_D.item():.3f} G={loss_G.item():.3f} LR={opt_G.param_groups[0]['lr']:.6f}")

        # Update Schedulers
        scheduler_G.step()
        scheduler_D.step()

        # Validation
        evaluate_and_show(epoch, G, val_set, device)

        # Save Checkpoint
        save_path = f"{CHECKPOINT_DIR}/checkpoint_epoch{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "sched_G": scheduler_G.state_dict(),
            "sched_D": scheduler_D.state_dict()
        }, save_path)
        print(f"Saved: {save_path}")

    print("Training complete.")
    generate_examples(G, val_set, device, n=10, epoch=EPOCHS)