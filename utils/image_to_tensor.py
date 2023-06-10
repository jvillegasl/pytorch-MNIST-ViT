from typing import Optional, Tuple
from PIL import Image

import torch
from torchvision.transforms.functional import pil_to_tensor, rgb_to_grayscale, invert


def image_to_tensor(
        image_path: str,
        invert_colors: bool = False,
        grayscale: bool = False,
        resize: Optional[Tuple[int, int]] = None
):
    image = Image.open(image_path)
    image = image.convert('RGB')
    if resize is not None:
        image = image.resize((28, 28))

    tensor = pil_to_tensor(image)

    if grayscale:
        tensor = rgb_to_grayscale(tensor)

    if invert_colors:
        tensor = invert(tensor)

    tensor = tensor.to(dtype=torch.float32)
    tensor = tensor / 255.0

    return tensor
