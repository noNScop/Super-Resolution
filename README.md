# Super Resolution

This repository contains implementations, training routines, and evaluation scripts for various **Single Image Super-Resolution (SISR)** models.

## Overview

All experiments are conducted using the **DIV2K** dataset.

1. **Baseline** - Bicubic Interpolation - Serves as a simple, traditional baseline to compare deep learning-based super-resolution models.

2. **FSRCNN** - Fast Super-Resolution Convolutional Neural Network - An enhanced version of SRCNN described in this [paper](https://arxiv.org/pdf/1608.00367). FSRCNN achieves better performance than bicubic interpolation while being faster and more efficient.

3. **SRGAN** - Super-Resolution Generative Adversarial Network - Described in this [paper](https://arxiv.org/pdf/1609.04802). This model also includes **SRResNet**, which is used as the generator in **SRGAN** but was previously state-of-the-art in **PSNR**-focused super-resolution tasks

4. **EDSR** - Enhanced Deep Super-Resolution Network - Based on [paper](https://arxiv.org/pdf/1707.02921), **EDSR** modifies **SRResNet** by removing **batch normalization** layers, which were found to degrade performance in super-resolution tasks. It also increases model capacity, with **~43M** parameters compared to SRResNet’s **~1.5M**.



5. **RCAN** - TODO

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