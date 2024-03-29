
"""
Introduces camera noise effects.

This module systematically models the different noise sources present in modern
digital cameras.

The implementation largely follows the work of
A. Carlson et al.: *Modeling Camera Effects to Improve Visual Learning from
Synthetic Data* (2018), `arXiv <https://arxiv.org/abs/1803.07721>`_

Typical usage
-------------

.. code:: python

    import stillleben as sl
    import torch

    # Input image (CHW)
    img = torch.rand(3, 1080, 1920)

    # Everything is much faster on GPU!
    img = img.cuda()

    # Apply noise
    img_with_noise = sl.camera_model.process_image(img)
"""

import torch
import math
import random

from . import profiling

__all__ = [
    'chromatic_aberration',
    'blur',
    'exposure',
    'noise',
    'color_jitter',
    'process_deterministic',
    'process_image',
]

@profiling.Timer('chromatic_aberration')
def chromatic_aberration(rgb, translations, scaling):
    """
    Introduces chromatic aberration effects.

    :param rgb: 3xHxW input RGB image
    :param translations: 3x2 translation tensor (tx, ty) for each of R, G, B
    :param scaling: [sr, sg, sb] scaling factor for each of R, G, B
    :return: 3xHxW output tensor
    """

    assert rgb.dim() == 3, "input tensor has invalid size {}".format(rgb.size())
    assert rgb.size(0) == 3, "input tensor has invalid size {}".format(rgb.size())

    transformations = torch.zeros(3, 2, 3)
    transformations[:,0,0] = scaling
    transformations[:,1,1] = scaling
    transformations[:,0:2,2] = translations

    grid = torch.nn.functional.affine_grid(transformations.to(rgb.device), (3,1,rgb.size(1),rgb.size(2)), align_corners=False)

    sampled = torch.nn.functional.grid_sample(rgb.unsqueeze(1), grid,
        mode='bilinear', padding_mode='reflection',
        align_corners=False
    )

    return sampled[:,0]


def _gaussian(sigma):
    # Set these to whatever you want for your gaussian filter
    kernel_size = 5

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)

    return gaussian_kernel

@profiling.Timer('blur')
def blur(rgb, sigma):
    """
    Applies Gaussian blur

    :param rgb: 3xHxW input RGB image
    :param sigma: Standard deviation of the Gaussian
    :return: 3xHxW output RGB image
    """

    return torch.nn.functional.conv2d(
        rgb.unsqueeze(1), _gaussian(sigma).to(rgb.device),
        padding=2, # needs to be filter_size/2 for equal output size
    )[:,0]

@profiling.Timer('exposure')
def exposure(rgb, deltaS):
    """
    Re-exposes the image

    :param rgb: 3xHxW input RGB image, float [0-1]
    :param deltaS: exposure shift (usually -1 to 1)
    :return: 3xHxW output image
    """

    return 1.0 / (1.0 + math.exp(deltaS) * (1.0 / (rgb + 0.0001) - 1.0))

@profiling.Timer('noise')
def noise(rgb, a, b):
    """
    Applies additional Poissonian-Gaussian noise

    :param rgb: 3xHxW input RGB image, float [0-1]
    :param a: variance factor of the signal-dependant noise (var = a*y(x))
    :param b: standard deviation of the signal-independant noise
    :return: modified image

    See Foi, Alessandro, et al. (2008):
    `Practical Poissonian-Gaussian noise modeling and fitting for single-image
    raw-data`,
    `PDF <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.164.1943&rep=rep1&type=pdf>`_
    """

    if a > 0.0:
        chi = 1.0 / a
        poisson_part = torch.poisson(chi * rgb) / chi
    else:
        poisson_part = rgb

    if b > 0.0:
        gaussian_part = torch.empty_like(rgb).normal_(std=b)
    else:
        gaussian_part = torch.zeros_like(rgb)

    return (poisson_part + gaussian_part).clamp_(0.0, 1.0)

def color_jitter(tensor_img, hue_shift):
    """
    Hue shift.

    :param tensor_img: 3xHxW RGB float tensor [0,1]
    :param hue_shift: hue shift to apply (-0.5 to 0.5)
    :return: jittered image
    """

    assert tensor_img.size(0) == 3
    height = tensor_img.size(1)
    width = tensor_img.size(2)

    R = tensor_img[0]
    G = tensor_img[1]
    B = tensor_img[2]

    M, Mi = tensor_img.max(dim=0)
    m, mi = tensor_img.min(dim=0)

    C = M - m

    H = tensor_img.new_empty((4, height, width))
    H[0] = 0.0
    H[1] = (G-B) / C + 0.0
    H[2] = (B-R) / C + 2.0
    H[3] = (R-G) / C + 4.0

    case = Mi + 1
    case[C == 0] = 0

    HSV = torch.empty_like(tensor_img)
    HSV[0] = 60.0 * H.gather(0, case.unsqueeze(0))[0]
    HSV[0][HSV[0] < 0] += 360.0

    HSV[1] = (M - m) / M
    HSV[1][M == 0] = 0.0
    HSV[2] = M

    # Apply jitter
    HSV[0] = HSV[0] + hue_shift * 360.0
    HSV[0][HSV[0] < 0] += 360.0
    HSV[0][HSV[0] > 360.0] -= 360.0

    # convert back to RGB
    HSV[0] /= 60.0
    X = C * (1.0 - (HSV[0].fmod(2.0) - 1).abs())

    order_case = HSV[0].view(-1).long().clamp_(0,5)
    order = torch.LongTensor([
        [0,1,2], [1,0,2], [2,0,1], [2,1,0], [1,2,0], [0,2,1]
    ]).to(HSV.device)
    selected_order = order[order_case].view(height, width, 3).permute(2, 0, 1)

    CX0 = torch.stack((C, X, torch.zeros_like(C)))

    RGB = CX0.gather(0, selected_order)
    RGB += m.unsqueeze(0)

    return RGB

def process_deterministic(rgb,
                          chromatic_translation, chromatic_scaling,
                          blur_sigma,
                          exposure_deltaS,
                          do_noise, noise_a, noise_b,
                          hue_shift):
    """
    Process image with given noise model parameters.
    """

    assert rgb.dim() == 3
    assert rgb.size(0) == 3

    rgb = chromatic_aberration(rgb,
        translations=chromatic_translation,
        scaling=chromatic_scaling
    )

    if blur_sigma > 0.0:
        rgb = blur(rgb, sigma=blur_sigma)

    rgb = exposure(rgb, deltaS=exposure_deltaS)

    if do_noise:
        rgb = noise(rgb, a=noise_a, b=noise_b)

    rgb.clamp_(0.0, 1.0)

    rgb = color_jitter(rgb, hue_shift)

    # The sensor noise we see has effect on more than one pixel
    # => apply a post-blur
    rgb = blur(rgb, sigma=0.4)

    rgb.clamp_(0.0, 1.0)

    return rgb

@profiling.Timer('camera_model.process_image')
def process_image(rgb):
    """
    Process image with random noise parameters.
    """

    assert rgb.dim() == 3
    assert rgb.size(0) == 3

    hue_jitter = 0.05

    return process_deterministic(
        rgb,

        chromatic_translation=torch.empty(3,2).uniform_(-0.002, 0.002),
        chromatic_scaling=torch.empty(3).uniform_(0.998, 1.002),

        blur_sigma=random.uniform(0.0, 3.0) if random.random() > 0.3 else 0.0,

        exposure_deltaS=random.uniform(-2, 1.2),

        do_noise=random.random() > 0.3,
        noise_a=random.random() * 0.04,
        noise_b=random.random() * 0.02,

        hue_shift=random.uniform(-hue_jitter, hue_jitter),
    )
