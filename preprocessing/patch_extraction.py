import numpy as np


def extract_patches(
    image,
    segmentation,
    patch_size=(64, 64, 64),
    stride=(64, 64, 64)
):
    """
    Extract 3D patches from a stacked MRI volume and its
    corresponding segmentation mask.

    image shape:
        (4, X, Y, Z)

    segmentation shape:
        (X, Y, Z)
    """

    px, py, pz = patch_size
    sx, sy, sz = stride

    _, x, y, z = image.shape

    image_patches = []
    segmentation_patches = []

    for i in range(0, x - px + 1, sx):
        for j in range(0, y - py + 1, sy):
            for k in range(0, z - pz + 1, sz):

                image_patch = image[
                    :,
                    i:i + px,
                    j:j + py,
                    k:k + pz
                ]

                segmentation_patch = segmentation[
                    i:i + px,
                    j:j + py,
                    k:k + pz
                ]

                image_patches.append(image_patch)
                segmentation_patches.append(segmentation_patch)

    return (
        np.array(image_patches),
        np.array(segmentation_patches)
    )