import matplotlib.pyplot as plt

from config import FIGURE_SIZE, MODALITIES


def show_modalities(stacked):

    slice_number = stacked.shape[3] // 2

    fig, axes = plt.subplots(
        1,
        len(MODALITIES),
        figsize=FIGURE_SIZE
    )

    for i, modality in enumerate(MODALITIES):

        axes[i].imshow(
            stacked[i, :, :, slice_number],
            cmap="gray"
        )

        axes[i].set_title(modality.upper())
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()