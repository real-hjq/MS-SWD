# MS-SWD
This repository provides the official PyTorch implementation of the paper [Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures](https://arxiv.org/abs/2407.10181), accepted at ECCV 2024.

---
# Requirements
- Python >= 3.7
- PyTorch >= 1.8

# Installation
Clone the repository:
```bash
git clone https://github.com/real-hjq/MS-SWD.git
cd MS-SWD
```

# Usage
Python API:
```python
from MS_SWD import MS_SWD

msswd_model = MS_SWD(num_scale=5, num_proj=128)
# X: (N, C, H, W)
# Y: (N, C, H, W)
distance = msswd_model(X, Y)
# distance : (N,)
```
Command line:
```bash
python MS_SWD.py --img1 <img1_path> --img2 <img2_path>
```

# Learned MS-SWD
A learned version of MS-SWD is available in [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).
```python
import pyiqa
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
msswd_model = pyiqa.create_metric('msswd', device=device)
```

# Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{he2024ms-swd,
  title={Multiscale Sliced {Wasserstein} Distances as Perceptual Color Difference Measures},
  author={He, Jiaqi and Wang, Zhihua and Wang, Leon and Liu, Tsein-I and Fang, Yuming and Sun, Qilin and Ma, Kede},
  booktitle={European Conference on Computer Vision},
  pages={1--18},
  year={2024},
  url={http://arxiv.org/abs/2407.10181}
}
```
# Acknowledgements
Part of this implementation is adapted from [GPDM](https://github.com/ariel415el/GPDM). The `srgb2lab` conversion code is adapted from `flip_loss.py` in [ꟻLIP](https://github.com/NVlabs/flip). We sincerely thank the authors for making their excellent work publicly available.
