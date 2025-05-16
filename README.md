# Super Resolution

This repository contains implementations, training routines, and evaluation scripts for various **Single Image Super-Resolution (SISR)** models.

## Overview

All experiments are conducted using the **DIV2K** dataset.

1. **Baseline** - Bicubic Interpolation - Serves as a simple, traditional baseline to compare deep learning-based super-resolution models.

2. **FSRCNN** - Fast Super-Resolution Convolutional Neural Network - An enhanced version of SRCNN described in this [paper](https://arxiv.org/pdf/1608.00367). FSRCNN achieves better performance than bicubic interpolation while being faster and more efficient.

3. **SRGAN** - Super-Resolution Generative Adversarial Network - Described in this [paper](https://arxiv.org/pdf/1609.04802). This model also includes **SRResNet**, which is used as the generator in **SRGAN** but was previously state-of-the-art in **PSNR**-focused super-resolution tasks. Due to instabilities during training, I was only able to train **SRGAN** for **X2 scaling**.

4. **RCAN** - Residual Channel Attention Network - A state-of-the-art deep learning model for single image super-resolution described in this [paper](https://arxiv.org/abs/1807.02758). RCAN leverages residual blocks and channel attention mechanisms to effectively enhance high-frequency details and achieve superior performance in high-scale image upscaling tasks. It demonstrates remarkable results, especially for higher scaling factors such as **X4** and **X8**.

## CLI Interface

This repository also includes a lightweight terminal interface that allows users to interactively:

- Choose a trained super-resolution model
- Select the upscaling factor (e.g., ×2, ×4, ×8)
- Provide a path to an input image
- Run the model and see where the output is saved

### To launch the terminal app:
```bash
python main.py
```
**Note**: This command should be executed from within the SR_app directory.

The upscaled output image will be saved in the output folder inside the same directory.

## Results
TODO

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.