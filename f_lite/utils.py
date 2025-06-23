import torch
from typing import Union, List
from PIL import Image
from torchvision.utils import make_grid

@torch.no_grad()
def make_image_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    **kwargs,
) -> None:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)