import os
import math
import torch
from PIL import Image
import torch.nn as nn
from datetime import datetime
from torchvision.transforms import v2

class BicubicInterpolation:
    def __init__(self, scale: int):
        self.name = "Bicubic"
        self.scale = scale

    def __call__(self, image: str):
        image = Image.open(image).convert("RGB")
        scaled_image = image.resize((self.scale * image.width, self.scale * image.height), Image.Resampling.BICUBIC)
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaled_image.save(f"output/{timestamp}_X{self.scale}.png")

class FSRCNN(nn.Module):
    def __init__(self, n: int, d: int = 56, s: int = 12, m: int = 4):
        """
        Args:
            d: feature dimension
            s: shrinking dimension
            m: mapping layers
            n: scaling factor
        """
        super().__init__()
        self.name = "FSRCNN"
        self.scale = n

        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: x/255.0)
        ])
        
        self.model = nn.Sequential(
            self._conv(3, d, 5),
            self._conv(d, s, 1)
        )

        for _ in range(m):
            self.model.append(self._conv(s, s, 3))

        self.model.append(self._conv(s, d, 1))

        # Ensure the output image is exactly n times bigger than the input
        if n <= 9:
            padding = (9 - n + 1) // 2
            output_padding = (9 - n) % 2
        else:
            for i in range(n):
                padding = i - n + 9
                if padding % 2 == 0 and padding >= 0:
                    output_padding = i
                    break
        
        self.model.append(nn.ConvTranspose2d(d, 3, 9, stride=n, padding=padding, output_padding=output_padding))

    def forward(self, image: str):
        image = Image.open(image).convert("RGB")
        inp = self.transform(image)
        out = self.model(inp).clamp(0.0, 1.0) * 255.0
        img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"output/{timestamp}_X{self.scale}.png")
        
    def _conv(self, ni, nf, ks):
        return nn.Sequential(
            nn.Conv2d(ni, nf, ks, padding='same'),
            nn.PReLU()
        )
    
class ResBlockEDSRLight(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding='same'),
                nn.PReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding='same'),
            )

        def forward(self, x):
            return x + self.block(x)

class EDSR_Light(nn.Module):
    def __init__(self, n: int):
        """
        Args:
            n: scaling factor
        """
        super().__init__()
        self.name = "EDSR_Light"
        self.scale = n

        self.DIV2K_RGB = torch.tensor([0.44882884613943946, 0.43713809810624193, 0.4040371984052683], device='cpu')
        self.transform = v2.Compose([
            v2.PILToTensor(),
            v2.Lambda(lambda x: (x / 255.0) - self.DIV2K_RGB[:, None, None])
        ])

        self.expand = nn.Sequential(
            nn.Conv2d(3, 64, 9, stride=1, padding='same'),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential()
        for _ in range(16):
            self.residual_blocks.append(ResBlockEDSRLight())

        self.residual_blocks.append(nn.Conv2d(64, 64, 3, stride=1, padding='same'))

        self.upscaling_head = nn.Sequential()
        for _ in range(int(math.log2(n))):
            self.upscaling_head.append(nn.Conv2d(64, 4*64, 3, stride=1, padding='same'))
            self.upscaling_head.append(nn.PixelShuffle(2))
            self.upscaling_head.append(nn.PReLU())
            
        self.upscaling_head.append(nn.Conv2d(64, 3, 9, stride=1, padding='same'))

    def forward(self, image: str):
        image = Image.open(image).convert("RGB")
        inp = self.transform(image)
        x = self.expand(inp)
        out = (self.upscaling_head(self.residual_blocks(x) + x) + self.DIV2K_RGB[:, None, None]).clamp(0.0, 1.0) * 255.0
        img = Image.fromarray(out.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img.save(f"output/{timestamp}_X{self.scale}.png")