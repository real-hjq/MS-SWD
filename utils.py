import sys
import torch


class GaussianPyramid(torch.nn.Module):
    def __init__(self, num_scale=5):
        super(GaussianPyramid, self).__init__()
        self.num_scale = num_scale

    def gauss_kernel(self, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        # Change kernel to PyTorch: (channels, 1, size, size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel

    def downsample(self, x):
        # Downsample along image (H,W)
        # Takes every 2 pixels: output (H, W) = input (H/2, W/2)
        return x[:, :, ::2, ::2]

    def conv_gauss(self, img, kernel):
        # Assume input of img is of shape: (N, C, H, W)
        if len(img.shape) != 4:
            raise IndexError('Expected input tensor to be of shape: (N, C, H, W) but got: ' + str(img.shape))
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        img = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return img

    def forward(self, img):
        kernel = self.gauss_kernel()
        kernel = kernel.to(img.device)
        img_pyramid = [img]
        for n in range(self.num_scale-1):
            img = self.conv_gauss(img, kernel)
            img = self.downsample(img)
            img_pyramid.append(img)
        return img_pyramid


def color_space_transform(input_color, fromSpace2toSpace):
    """
    Transforms inputs between different color spaces
    :param input_color: tensor of colors to transform (with NxCxHxW layout)
    :param fromSpace2toSpace: string describing transform
    :return: transformed tensor (with NxCxHxW layout)
    """
    dim = input_color.size()
    device = input_color.device

    # Assume D65 standard illuminant
    reference_illuminant = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]]).to(device)
    inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).to(device)

    if fromSpace2toSpace == "srgb2linrgb":
        limit = 0.04045
        transformed_color = torch.where(
            input_color > limit,
            torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
            input_color / 12.92
        )  # clamp to stabilize training

    elif fromSpace2toSpace == "linrgb2srgb":
        limit = 0.0031308
        transformed_color = torch.where(
            input_color > limit,
            1.055 * torch.pow(torch.clamp(input_color, min=limit), (1.0 / 2.4)) - 0.055,
            12.92 * input_color
        )

    elif fromSpace2toSpace in ["linrgb2xyz", "xyz2linrgb"]:
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        if fromSpace2toSpace == "linrgb2xyz":
            a11 = 10135552 / 24577794
            a12 = 8788810 / 24577794
            a13 = 4435075 / 24577794
            a21 = 2613072 / 12288897
            a22 = 8788810 / 12288897
            a23 = 887015 / 12288897
            a31 = 1425312 / 73733382
            a32 = 8788810 / 73733382
            a33 = 70074185 / 73733382
        else:
            # Constants found by taking the inverse of the matrix
            # defined by the constants for linrgb2xyz
            a11 = 3.241003275
            a12 = -1.537398934
            a13 = -0.498615861
            a21 = -0.969224334
            a22 = 1.875930071
            a23 = 0.041554224
            a31 = 0.055639423
            a32 = -0.204011202
            a33 = 1.057148933
        A = torch.Tensor([[a11, a12, a13],
                          [a21, a22, a23],
                          [a31, a32, a33]])

        input_color = input_color.view(dim[0], dim[1], dim[2] * dim[3])  # NC(HW)

        transformed_color = torch.matmul(A.to(device), input_color)
        transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])

    elif fromSpace2toSpace == "xyz2ycxcz":
        input_color = torch.mul(input_color, inv_reference_illuminant)
        y = 116 * input_color[:, 1:2, :, :] - 16
        cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
        transformed_color = torch.cat((y, cx, cz), 1)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        cx = input_color[:, 1:2, :, :] / 500
        cz = input_color[:, 2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = torch.cat((x, y, z), 1)

        transformed_color = torch.mul(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        input_color = torch.mul(input_color, inv_reference_illuminant)
        delta = 6 / 29
        delta_square = delta * delta
        delta_cube = delta * delta_square
        factor = 1 / (3 * delta_square)

        clamped_term = torch.pow(torch.clamp(input_color, min=delta_cube), 1.0 / 3.0).to(dtype=input_color.dtype)
        div = (factor * input_color + (4 / 29)).to(dtype=input_color.dtype)
        input_color = torch.where(input_color > delta_cube, clamped_term, div)  # clamp to stabilize training

        L = 116 * input_color[:, 1:2, :, :] - 16
        a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
        b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])

        transformed_color = torch.cat((L, a, b), 1)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[:, 0:1, :, :] + 16) / 116
        a = input_color[:, 1:2, :, :] / 500
        b = input_color[:, 2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = torch.cat((x, y, z), 1)
        delta = 6 / 29
        delta_square = delta * delta
        factor = 3 * delta_square
        xyz = torch.where(xyz > delta, torch.pow(xyz, 3), factor * (xyz - 4 / 29))

        transformed_color = torch.mul(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)

    return transformed_color
