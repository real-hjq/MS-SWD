import torch
import torch.nn.functional as F
from utils import GaussianPyramid
from utils import color_space_transform


class MS_SWD(torch.nn.Module):
    def __init__(self, num_scale=5, num_proj=128, patch_size=11, stride=1, c=3):
        super(MS_SWD, self).__init__()
        self.num_scale = num_scale
        self.num_proj = num_proj
        self.patch_size = patch_size
        self.stride = stride
        self.c = c
        self.sample_projections()
        self.gaussian_pyramid = GaussianPyramid(num_scale)

    def sample_projections(self):
        # Sample random normalized projections
        rand = torch.randn(self.num_proj, self.c*self.patch_size**2)
        rand = rand / torch.norm(rand, dim=1, keepdim=True)  # normalize to unit directions
        self.rand = rand.reshape(self.num_proj, self.c, self.patch_size, self.patch_size)

    def forward_once(self, x, y, reset_projections=True):
        if reset_projections:
            self.sample_projections()
        self.rand = self.rand.to(x.device)

        # Project patches
        pad_num = self.patch_size // 2
        x = F.pad(x, pad=(pad_num, pad_num, pad_num, pad_num), mode='reflect')
        y = F.pad(y, pad=(pad_num, pad_num, pad_num, pad_num), mode='reflect')
        projx = F.conv2d(x, self.rand, stride=self.stride).reshape(x.shape[0], self.num_proj, -1)
        projy = F.conv2d(y, self.rand, stride=self.stride).reshape(y.shape[0], self.num_proj, -1)
        # Sort and compute L1 loss
        projx, _ = torch.sort(projx, dim=2)
        projy, _ = torch.sort(projy, dim=2)
        swd = torch.abs(projx - projy)
        return torch.mean(swd, dim=[1, 2])

    def forward(self, x, y):
        ms_swd = 0
        # Build Gaussian pyramids
        x_pyramid = self.gaussian_pyramid(x)
        y_pyramid = self.gaussian_pyramid(y)
        for n in range(self.num_scale):
            # Image preprocessing
            x_single = color_space_transform(x_pyramid[n], 'srgb2lab')
            y_single = color_space_transform(y_pyramid[n], 'srgb2lab')
            swd = self.forward_once(x_single, y_single)
            ms_swd = ms_swd + swd
        ms_swd = ms_swd/self.num_scale
        return ms_swd


def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.Resize(256)(image)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


if __name__ == "__main__":
    import argparse
    from PIL import Image
    from torchvision import transforms

    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, default='./images/non-aligned_01.png')
    parser.add_argument('--img2', type=str, default='./images/non-aligned_02.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MS_SWD(num_scale=5, num_proj=128).to(device)

    img1 = prepare_image(Image.open(args.img1).convert("RGB")).to(device)
    img2 = prepare_image(Image.open(args.img2).convert("RGB")).to(device)
    assert img1.shape == img2.shape

    distance = model(img1, img2)

    print('distance: %.3f' % distance.item())

