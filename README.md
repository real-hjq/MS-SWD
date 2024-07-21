# MS-SWD
This is the repository of paper [Multiscale Sliced Wasserstein Distances as Perceptual Color Difference Measures](http://arxiv.org/abs/2407.10181).

---
# Requirement
- Python>=2.6
- Pytorch>=2.0

# Useage
```python
from MS-SWD import MS-SWD
model = MS_SWD(num_scale=5, num_proj=128)
# X: (N,C,H,W)
# Y: (N,C,H,W)
distance = Model(X, Y)
```
or
```c
git clone https://github.com/real-hjq/MS-SWD
cd MS-SWD

python MS_SWD.py --img1 <img1_path> --img2 <img2_path>
```

# Citation
```c
@inproceedings{he2024ms-swd,
  title={Multiscale Sliced {Wasserstein} Distances as Perceptual Color Difference Measures},
  author={He, Jiaqi and Wang, Zhihua and Wang, Leon and Liu, Tsein-I and Fang, Yuming and Sun, Qilin and Ma, Kede},
  booktitle={European Conference on Computer Vision},
  pages={1--18},
  year={2024},
  url={http://arxiv.org/abs/2407.10181}
}
```
