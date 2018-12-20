
"""
Introduces camera noise effects

Largely after:
A. Carlson et al.: Modeling Camera Effects to Improve Visual Learning from
  Synthetic Data (2018)
"""

import torch
import math
import random

from . import profiling

@profiling.Timer('chromatic_aberration')
def chromatic_aberration(rgb, translations, scaling):
    """
    Introduces chromatic aberration effects.

    Args:
        rgb (tensor): 3xHxW input RGB image
        translation (tensor): 3x2 translation tensor (tx, ty) for each of R, G, B
        scaling (tensor): [sr, sg, sb] scaling factor for each of R, G, B

    Returns:
        tensor: 3xHxW output tensor
    """

    assert rgb.dim() == 3, "input tensor has invalid size {}".format(rgb.size())
    assert rgb.size(0) == 3, "input tensor has invalid size {}".format(rgb.size())

    transformations = torch.zeros(3, 2, 3)
    transformations[:,0,0] = scaling
    transformations[:,1,1] = scaling
    transformations[:,0:2,2] = translations

    grid = torch.nn.functional.affine_grid(transformations, (3,1,rgb.size(1),rgb.size(2)))

    sampled = torch.nn.functional.grid_sample(rgb.unsqueeze(1), grid,
        mode='bilinear', padding_mode='reflection'
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

    Args:
        rgb (tensor): 3xHxW input RGB image
        sigma (float): Standard deviation of the Gaussian

    Returns:
        tensor: 3xHxW output RGB image
    """

    return torch.nn.functional.conv2d(rgb.unsqueeze(1), _gaussian(sigma),
        padding=2, # needs to be filter_size/2 for equal output size
    )[:,0]

@profiling.Timer('exposure')
def exposure(rgb, deltaS):
    """
    Re-exposes the image

    Args:
        rgb (tensor): 3xHxW input RGB image, float [0-1]
        deltaS (float): exposure shift (usually 0 to 2)

    Returns:
        tensor: 3xHxW output image
    """

    return 1.0 / (1.0 + deltaS * (1.0 / (rgb + 0.0001) - 1.0))

@profiling.Timer('noise')
def noise(rgb, a, b):
    """
    Applies additional noise, following

    "Practical Poissonian-Gaussian noise modeling and fitting for single-image
    raw-data."
    Foi, Alessandro, et al. (2008)

    Args:
        rgb (tensor): 3xHxW input RGB image, float [0-1]
        a (float): variance factor of the signal-dependant noise (var = a*y(x))
        b (float): standard deviation of the signal-independant noise
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

@profiling.Timer('camera_model.process_image')
def process_image(rgb):

    rgb = chromatic_aberration(rgb,
        translations=torch.empty(3,2).uniform_(-0.002, 0.002),
        scaling=torch.empty(3).uniform_(0.998, 1.002)
    )

    rgb = blur(rgb, sigma=random.uniform(0.0, 3.0))

    rgb = exposure(rgb, deltaS=random.uniform(0.001, 2.0))

    rgb = noise(rgb, a=0.001, b=0.01)

    return rgb


