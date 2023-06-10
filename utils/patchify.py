from torch import Tensor


def patchify(images: Tensor, patches_per_side: int) -> Tensor:
    """
    Arguments:
        images: Tensor, shape ``[batch_size, color_channels, height, width]``

    Returns:
        patches: Tensor, shape ``[batch_size, n_patches, color_channels, patch_height, patch_width]``
    """
    n, c, h, w = images.shape

    assert h == w, 'Patchify method is implemented for square images only'
    assert w % patches_per_side == 0, 'Decimal size patches cannot be generated (size = (width or height) / n_patches)'

    patch_size = int(w / patches_per_side)

    patches = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size)

    patches = patches.reshape(n, c, -1, patch_size,
                              patch_size).permute(0, 2, 1, 3, 4)

    return patches
