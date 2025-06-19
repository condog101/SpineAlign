import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
from typing import Tuple


def adjust_hue(rgb: torch.Tensor, hue_factor: float) -> torch.Tensor:
    """
    Adjust hue of RGB colors
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        hue_factor: How much to adjust the hue. Can be negative.
                   -0.5 to 0.5 maps to -180° to 180° degrees.
    Returns:
        Adjusted RGB tensor of shape (N, 3)
    """
    return F.adjust_hue(rgb, hue_factor)


def adjust_saturation(rgb: torch.Tensor, saturation_factor: float) -> torch.Tensor:
    """
    Adjust saturation of RGB colors
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        saturation_factor: How much to adjust the saturation.
                          0 = grayscale, 1 = original, 2 = double saturation
    Returns:
        Adjusted RGB tensor of shape (N, 3)
    """
    return F.adjust_saturation(rgb, saturation_factor)


def adjust_brightness(rgb: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    """
    Adjust brightness of RGB colors
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        brightness_factor: How much to adjust the brightness.
                         0 = black, 1 = original, 2 = double brightness
    Returns:
        Adjusted RGB tensor of shape (N, 3)
    """
    return F.adjust_brightness(rgb, brightness_factor)


def adjust_contrast(rgb: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    """
    Adjust contrast of RGB colors
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        contrast_factor: How much to adjust the contrast.
                        0 = gray image, 1 = original, 2 = double contrast
    Returns:
        Adjusted RGB tensor of shape (N, 3)
    """
    return F.adjust_contrast(rgb, contrast_factor)


def adjust_gamma(rgb: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Apply gamma correction to RGB colors
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        gamma: Non-negative real number. gamma > 1 makes shadows darker,
               gamma < 1 makes dark regions lighter.
    Returns:
        Gamma-corrected RGB tensor of shape (N, 3)
    """
    return F.adjust_gamma(rgb, gamma)


def random_color_jitter(rgb: torch.Tensor,
                        brightness: float = 0.2,
                        contrast: float = 0.2,
                        saturation: float = 0.2,
                        hue: float = 0.1) -> torch.Tensor:
    """
    Randomly adjust color properties
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        brightness: Maximum brightness adjustment
        contrast: Maximum contrast adjustment
        saturation: Maximum saturation adjustment
        hue: Maximum hue adjustment (-0.5 to 0.5)
    Returns:
        Randomly adjusted RGB tensor of shape (N, 3)
    """
    jitter = T.ColorJitter(brightness=brightness,
                           contrast=contrast,
                           saturation=saturation,
                           hue=hue)
    return jitter(rgb)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to HSV
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
    Returns:
        HSV tensor of shape (N, 3) with values in [0, 1]
    """
    return F.rgb_to_hsv(rgb)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV to RGB
    Args:
        hsv: tensor of shape (N, 3) with values in [0, 1]
    Returns:
        RGB tensor of shape (N, 3) with values in [0, 1]
    """
    return F.hsv_to_rgb(hsv)


def normalize_rgb(rgb: torch.Tensor,
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Normalize RGB values using mean and standard deviation
    Args:
        rgb: tensor of shape (N, 3) with values in [0, 1]
        mean: Means for each channel
        std: Standard deviations for each channel
    Returns:
        Normalized RGB tensor of shape (N, 3)
    """
    return F.normalize(rgb, mean=mean, std=std)


def expand_to_im_size(rgb: torch.Tensor):
    rgb = rgb.unsqueeze(-1).unsqueeze(-1)
    return rgb


def reduce_from_im_size(rgb: torch.Tensor):
    rgb = rgb.squeeze(-1).squeeze(-1)
    return rgb
