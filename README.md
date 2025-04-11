# Super Resolution

This repository contains implementations, training routines, and evaluation scripts for various **Single Image Super-Resolution (SISR)** models.

## Overview

All experiments are conducted using the **DIV2K** dataset.

1. **Baseline** - Bicubic Interpolation - Serves as a simple, traditional baseline to compare deep learning-based super-resolution models.

2. **FSRCNN** - Fast Super-Resolution Convolutional Neural Network - An enhanced version of SRCNN described in the [original paper](https://arxiv.org/pdf/1608.00367). FSRCNN achieves better performance than bicubic interpolation while being faster and more efficient.

3. **EDSR** - Enhanced Deep Super-Resolution Network - Described in this [paper](https://arxiv.org/pdf/1707.02921), EDSR is a powerful architecture that surpasses many earlier methods in both accuracy and visual quality. Due to hardware limitations, I trained a lighter version of the full EDSR (which has over **43M parameters**). This smaller model has around **1.5M parameters** and corresponds to the baseline variant discussed in the original paper. It is comparable in size to SRResNet, which EDSR is built upon. In fact, my implementation can be viewed as *SRResNet without batch normalization*.

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