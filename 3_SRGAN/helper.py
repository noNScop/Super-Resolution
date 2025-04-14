import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import v2
from torch.utils.data import Dataset
from torchmetrics.image import PeakSignalNoiseRatio
from torch.optim.lr_scheduler import StepLR
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

transform = v2.Compose([
    v2.PILToTensor(),
    v2.Lambda(lambda x: x / 255.0)
])

class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding='same'),
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding='same'),
                nn.BatchNorm2d(64)
            )

        def forward(self, x):
            return x + self.block(x)

class SRResNet(nn.Module):
    def __init__(self, n: int):
        """
        Args:
            n: scaling factor
        """
        super().__init__()
        
        self.expand = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding='same'),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential()
        for _ in range(16):
            self.residual_blocks.append(ResBlock())

        self.residual_blocks.append(nn.Conv2d(64, 64, 3, stride=1, padding='same'))
        self.residual_blocks.append(nn.BatchNorm2d(64))

        self.upscaling_head = nn.Sequential()
        for _ in range(int(math.log2(n))):
            self.upscaling_head.append(nn.Conv2d(64, 256, 3, stride=1, padding='same'))
            self.upscaling_head.append(nn.PixelShuffle(2))
            self.upscaling_head.append(nn.PReLU())
            
        self.upscaling_head.append(nn.Conv2d(64, 3, 9, stride=1, padding='same'))

    def forward(self, x):
        x = self.expand(x)
        return self.upscaling_head(self.residual_blocks(x) + x)
        
class SRResNet_Dataset(Dataset):
    def __init__(self, target_paths: list[str], scale: int, ram_limit_gb: float = 2.0):
        self.crop_size = scale * 48
        self.scale = scale

        self.input_transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: x / 255.0)
        ])
        
        self.target_transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: (2.0*x / 255.0) - 1)
        ])

        self.preloaded = {}
        self.paths = target_paths

        total_ram_used = 0
        for i, path in enumerate(tqdm(target_paths, desc="Preloading images")):
            img = Image.open(path).convert("RGB")
            total_ram_used += img.width * img.height * 3 / (1024 ** 3)  # ~size in GB

            if total_ram_used < ram_limit_gb:
                self.preloaded[i] = img
            else:
                break

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx in self.preloaded:
            target = self.preloaded[idx]
        else:
            target = Image.open(self.paths[idx]).convert("RGB")

        target = self.random_crop(target, self.crop_size)
        inp = target.resize((target.width // self.scale, target.height // self.scale), Image.BICUBIC)
            
        return self.input_transform(inp), self.target_transform(target)

    def random_crop(self, img, size):
        w, h = img.size
        if w < size or h < size:
            img = img.resize((size, size), Image.BICUBIC)
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return img.crop((x, y, x + size, y + size))

    def set_scale(self, scale: int):
        self.scale = scale

    def set_crop_size(self, crop_size: int):
        self.crop_size = crop_size
        
def calc_metrics(model: nn.Module, target_ds: list[str], scale: int):
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Lambda(lambda x: x / 255.0)
    ])

    psnr_acc = 0
    ssim_acc = 0
    lpips_acc = 0
    failed_lpips = 0

    for i in tqdm(range(len(target_ds)), leave=False):
        target_image = Image.open(target_ds[i]).convert("RGB")
        w, h = target_image.size

        w -= w % scale
        h -= h % scale
        target_image = target_image.crop((0, 0, w, h))
        
        lowres = target_image.resize((w // scale, h // scale), resample=Image.BICUBIC)
        input_tensor = transform(lowres).unsqueeze(0).to(device)
        target_tensor = transform(target_image).unsqueeze(0).to(device)

        with torch.inference_mode():
            # convert value range [-1,1] -> [0,1]
            sr = ((model(input_tensor)+1.0)/2.0).clamp(0.0, 1.0)

        psnr_acc += psnr(sr, target_tensor).item()
        ssim_acc += ssim(sr, target_tensor).item()
        
        # There are 2 images that cause lpips to fail
        try:
            x = lpips(sr, target_tensor).cpu().item()
            if np.isnan(x):
                failed_lpips += 1
                continue
                
            lpips_acc += x
        except:
            failed_lpips += 1

    lpips_acc /= len(target_ds) - failed_lpips
    psnr_acc /= len(target_ds)
    ssim_acc /= len(target_ds)
    return psnr_acc, ssim_acc, lpips_acc
    
def train_step(model, dataloader, optimizer, loss_fn):
    avg_psnr = 0
    avg_ssim = 0
    model.train()

    for batch, target in dataloader:
        batch, target = batch.to(device), target.to(device)
        
        logits = model(batch)
        loss = loss_fn(logits, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # convert value range [-1,1] -> [0,1]
        logits = ((logits+1.0)/2.0).clamp(0.0, 1.0)
        target = ((target+1.0)/2.0).clamp(0.0, 1.0)
        
        avg_psnr += psnr(logits, target).item()
        avg_ssim += ssim(logits, target).item()
        
    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)
    return avg_psnr, avg_ssim

def valid_step(model, dataloader, loss_fn):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    model.eval()

    with torch.inference_mode():
        for batch, target in dataloader:
            batch, target = batch.to(device), target.to(device)

            # convert [-1,1] -> [0,1]
            logits = ((model(batch)+1.0)/2.0).clamp(0.0, 1.0)
            target = ((target+1.0)/2.0).clamp(0.0, 1.0)
            
            avg_psnr += psnr(logits, target).item()
            avg_ssim += ssim(logits, target).item()
            avg_lpips += lpips(logits, target).item()

    avg_psnr /= len(dataloader)
    avg_ssim /= len(dataloader)
    avg_lpips /= len(dataloader)
 
    return avg_psnr, avg_ssim, avg_lpips

def train(model, train_dl, valid_dl, optimizer, loss_fn, epochs, start_checkpoint=None):
    os.makedirs('../tmp_model_checkpoints', exist_ok=True)
    counter = 0 # count epochs without printing training stats
    
    if start_checkpoint:
        start_epoch = start_checkpoint['epoch'] + 1
        best_psnr = start_checkpoint['best_psnr']
        best_ssim = start_checkpoint['best_ssim']
        best_lpips = start_checkpoint['best_lpips']
    else:
        start_epoch = 0
        best_psnr = 0
        best_ssim = 0
        best_lpips = float('inf')
        
    log_freq = (epochs - start_epoch) // 100 # how often to print stats when no progress is made
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        counter += 1
        train_psnr, train_ssim = train_step(
            model,
            train_dl,
            optimizer,
            loss_fn
        )

        valid_psnr, valid_ssim, valid_lpips = valid_step(
            model,
            valid_dl,
            loss_fn,
        )

        progress = False
        
        if valid_psnr > best_psnr:
            progress = True
            best_psnr = valid_psnr
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict':  model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best_psnr.pth')

        if valid_ssim > best_ssim:
            progress = True
            best_ssim = valid_ssim
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict':  model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best_ssim.pth')

        if valid_lpips < best_lpips:
            progress = True
            best_lpips = valid_lpips
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict':  model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best_lpips.pth')

        if epoch == epochs-1:
            checkpoint = {
                'epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': best_ssim,
                'best_lpips': best_lpips,
                'model_state_dict':  model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/last.pth')
            
        if progress or counter >= log_freq:
            counter = 0
            print(
                f"Epoch: {epoch+1} | "
                f"[train] PSNR: {train_psnr:.4f} | "
                f"[train] SSIM: {train_ssim:.4f} | "
                f"[valid] PSNR: {valid_psnr:.4f} | "
                f"[valid] SSIM: {valid_ssim:.4f} | "
                f"[valid] LPIPS: {valid_lpips:.4f}"
            )