import os
import math
import glob
import torch
import random
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import Adam
from torchinfo import summary
from torchvision.models import vgg19
from torchvision.transforms import v2
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).to(device)
vgg54 = vgg.features[:36]
vgg54.eval()

psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

transform = v2.Compose([
    v2.PILToTensor(),
    v2.Lambda(lambda x: x / 255.0)
])

HR_train_paths = sorted(glob.glob("../data/DIV2K_train_HR/*.png"))
X2_train_paths = sorted(glob.glob("../data/DIV2K_train_LR_bicubic/X2/*.png"))
X4_train_paths = sorted(glob.glob("../data/DIV2K_train_LR_bicubic/X4/*.png"))
X8_train_paths = sorted(glob.glob("../data/DIV2K_train_LR_bicubic/X8/*.png"))
X16_train_paths = sorted(glob.glob("../data/DIV2K_train_LR_bicubic/X16/*.png"))
X32_train_paths = sorted(glob.glob("../data/DIV2K_train_LR_bicubic/X32/*.png"))
X64_train_paths = sorted(glob.glob("../data/DIV2K_train_LR_bicubic/X64/*.png"))

HR_valid_paths = sorted(glob.glob("../data/DIV2K_valid_HR/*.png"))
X2_valid_paths = sorted(glob.glob("../data/DIV2K_valid_LR_bicubic/X2/*.png"))
X4_valid_paths = sorted(glob.glob("../data/DIV2K_valid_LR_bicubic/X4/*.png"))
X8_valid_paths = sorted(glob.glob("../data/DIV2K_valid_LR_bicubic/X8/*.png"))
X16_valid_paths = sorted(glob.glob("../data/DIV2K_valid_LR_bicubic/X16/*.png"))
X32_valid_paths = sorted(glob.glob("../data/DIV2K_valid_LR_bicubic/X32/*.png"))
X64_valid_paths = sorted(glob.glob("../data/DIV2K_valid_LR_bicubic/X64/*.png"))

class ResBlockSRRN(nn.Module):
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
            self.residual_blocks.append(ResBlockSRRN())

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

class ConvBlock(nn.Module):
    def __init__(self, ni: int, nf: int, ks: int, stride: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(ni, nf, ks, stride=stride, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.body = nn.Sequential(
            ConvBlock(64, 64, 3, 2),
            ConvBlock(64, 128, 3, 1),
            ConvBlock(128, 128, 3, 2),
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 256, 3, 2),
            ConvBlock(256, 512, 3, 1),
            ConvBlock(512, 512, 3, 2)
        )

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.model = nn.Sequential(
            self.expand,
            self.body,
            self.avgpool,
            self.head   
        )

    def forward(self, x):
        return self.model(x)

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

def train_generator(batch, target, generator, discriminator, optimizer, scaler, pixel_loss_fn):
    generator.train()
    discriminator.eval()
    
    optimizer.zero_grad(set_to_none=True)

    with autocast('cuda'):
        fake_imgs = generator(batch)
        # VGG MSE is scaled by 0.006 to put it in similar scale to MSE on images, Unlike in the paper I also add the pixel wise MSE, without it images are distorted
        loss = pixel_loss_fn(fake_imgs, target) + 0.006 * torch.mean((vgg54(target) - vgg54(fake_imgs))**2) - 0.001 * torch.mean(torch.log(discriminator(fake_imgs) + 1e-8))

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

def train_discriminator(LR, HR, generator, discriminator, optimizer, loss_fn):
    generator.eval()
    discriminator.train()
    
    with torch.inference_mode():
        fake_imgs = generator(LR[:len(LR)//2]).detach()
    real_imgs = HR[len(HR)//2:]
    
    batch = torch.cat([fake_imgs, real_imgs], dim=0).to(device)
    labels = torch.cat([
        torch.zeros(len(fake_imgs), device=device),  # fake = 0
        torch.ones(len(real_imgs), device=device)    # real = 1
    ], dim=0)[:, None]

    optimizer.zero_grad(set_to_none=True)

    probs = discriminator(batch)
    loss = loss_fn(probs, labels)

    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_step(generator, discriminator, dataloader, generator_opt, discriminator_opt, dis_loss_fn, pixel_loss_fn, generator_scaler):
    discriminator_loss = 0
    generator_loss = 0
    
    for batch, target in dataloader:
        batch, target = batch.to(device), target.to(device)
        discriminator_loss += train_discriminator(batch, target, generator, discriminator, discriminator_opt, dis_loss_fn)
        generator_loss += train_generator(batch, target, generator, discriminator, generator_opt, generator_scaler, pixel_loss_fn)
        
    discriminator_loss /= len(dataloader)
    generator_loss /= len(dataloader)
    return discriminator_loss, generator_loss

def valid_step(generator, discriminator, dataloader, dis_loss_fn, pixel_loss_fn):
    generator.eval()
    discriminator.eval()
    discriminator_loss = 0
    generator_loss = 0
    lpips_acc = 0

    with torch.inference_mode():
        for batch, target in dataloader:
            batch, target = batch.to(device), target.to(device)
            
            fake_imgs = generator(batch)
            lpips_acc += lpips(((fake_imgs+1.0)/2.0).clamp(0.0, 1.0), ((target+1.0)/2.0).clamp(0.0, 1.0))
            
            generator_loss += (pixel_loss_fn(fake_imgs, target) - 0.006 * torch.mean((vgg54(target) - vgg54(fake_imgs))**2) 
                               - 0.001 * torch.mean(torch.log(discriminator(fake_imgs) + 1e-8))).item()

            fake_imgs = fake_imgs[:len(target)//2]
            real_imgs = target[len(target)//2:]
            
            batch = torch.cat([fake_imgs, real_imgs], dim=0).to(device)
            labels = torch.cat([
                torch.zeros(len(fake_imgs), device=device),  # fake = 0
                torch.ones(len(real_imgs), device=device)    # real = 1
            ], dim=0)[:, None]

            probs = discriminator(batch)
            discriminator_loss += dis_loss_fn(probs, labels).item()
        
        discriminator_loss /= len(dataloader) 
        generator_loss /= len(dataloader)
        lpips_acc /= len(dataloader)


    return discriminator_loss, generator_loss, lpips_acc

def train(generator, discriminator, train_dl, valid_dl, generator_opt, discriminator_opt, generator_scheduler: StepLR, 
          discriminator_scheduler: StepLR, dis_loss_fn, pixel_loss_fn, epochs, start_checkpoint=None):
    os.makedirs('../tmp_model_checkpoints', exist_ok=True)
    counter = 0 # count epochs without printing training stats
    best_lpips= float('inf')
    generator_scaler = GradScaler('cuda')
    
    if start_checkpoint:
        start_epoch = start_checkpoint['epoch'] + 1
        best_lpips = start_checkpoint['lpips']
        generator_scaler.load_state_dict(start_checkpoint['generator_scaler_state_dict'])
    else:
        start_epoch = 0
        
    log_freq = (epochs - start_epoch) // 100 # how often to print stats when no progress is made
    
    for _ in tqdm(range(100), desc="Discriminator warm up"):
        for batch, target in train_dl:
            batch, target = batch.to(device), target.to(device)
            train_discriminator(batch, target, generator, discriminator, discriminator_opt, dis_loss_fn)
        
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs"):
        counter += 1
        train_d_loss, train_g_loss = train_step(
            generator,
            discriminator,
            train_dl,
            generator_opt,
            discriminator_opt,
            dis_loss_fn,
            pixel_loss_fn,
            generator_scaler
        )

        valid_d_loss, valid_g_loss, valid_lpips = valid_step(
            generator,
            discriminator,
            valid_dl,
            dis_loss_fn,
            pixel_loss_fn
        )

        discriminator_scheduler.step()
        generator_scheduler.step()

        progress = False
        
        if valid_lpips < best_lpips:
            progress = True
            best_lpips = valid_lpips
            checkpoint = {
                'epoch': epoch,
                'lpips': best_lpips,
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_opt.state_dict(),
                'generator_optimizer_state_dict': generator_opt.state_dict(),
                'discriminator_scheduler_state_dict': discriminator_scheduler.state_dict(),
                'generator_scheduler_state_dict': generator_scheduler.state_dict(),
                'generator_scaler_state_dict': generator_scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/best.pth')

        if epoch == epochs-1:
            checkpoint = {
                'epoch': epoch,
                'lpips': best_lpips,
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_opt.state_dict(),
                'generator_optimizer_state_dict': generator_opt.state_dict(),
                'discriminator_scheduler_state_dict': discriminator_scheduler.state_dict(),
                'generator_scheduler_state_dict': generator_scheduler.state_dict(),
                'generator_scaler_state_dict': generator_scaler.state_dict()
            }
            torch.save(checkpoint, f'../tmp_model_checkpoints/last.pth')
            
        if progress or counter >= log_freq:
            counter = 0
            print(
                f"Epoch: {epoch+1} | "
                f"learning rate: {generator_scheduler.get_last_lr()[0]:.6f} | "
                f"[train] generator: {train_g_loss:.4f} | "
                f"[train] discriminator: {train_d_loss:.4f} | "
                f"[valid] generator: {valid_g_loss:.4f} | "
                f"[valid] discriminator: {valid_d_loss:.4f} | "
                f"[valid] lpips: {valid_lpips:.4f}"
            )

        # Store intermediate results to see how training goes
        if (epoch + 350) % 400 == 0:
            inp = transform(Image.open(X8_valid_paths[4])).to(device)
            with torch.inference_mode():
                out = (((generator(inp[None])+1.0)/2.0).clamp(0.0, 1.0) * 255.0).squeeze()
            
            img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
            os.makedirs('./mid_training_images/', exist_ok=True)
            img.save(f'./mid_training_images/{epoch}_img1.png')
            
            inp = transform(Image.open('/home/noNScop/Desktop/test.png').convert("RGB")).to(device)
            with torch.inference_mode():
                out = (((generator(inp[None])+1.0)/2.0).clamp(0.0, 1.0) * 255.0).squeeze()
            
            img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
            img.save(f'./mid_training_images/{epoch}_img2.png')
            
            inp = transform(Image.open('/home/noNScop/Desktop/test2.png').convert("RGB")).to(device)
            with torch.inference_mode():
                out = (((generator(inp[None])+1.0)/2.0).clamp(0.0, 1.0) * 255.0).squeeze()
            
            img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
            img.save(f'./mid_training_images/{epoch}_img3.png')