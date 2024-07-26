# MS-SWD
This is the repository of paper [Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures](http://arxiv.org/abs/2407.10181).

---
# Requirement
- Python>=3.0
- Pytorch>=2.0

# Useage
```python
from MS-SWD import MS_SWD
model = MS_SWD(num_scale=5, num_proj=128)
# X: (N,C,H,W)
# Y: (N,C,H,W)
distance = model(X, Y)
```
or
```c
git clone https://github.com/real-hjq/MS-SWD
cd MS-SWD

python MS_SWD.py --img1 <img1_path> --img2 <img2_path>
```

# Citation
```
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
Part of the code is borrowed from [GPDM](https://github.com/ariel415el/GPDM), and srgb2lab comes from flip_loss.py in [ꟻLIP](https://github.com/NVlabs/flip). Sincerely thank them for their wonderful works.
