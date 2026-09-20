import numpy as np

from config import MODALITIES


def stack_modalities(patient):

    stacked = np.stack(

        [
            patient["modalities"][m]
            for m in MODALITIES
        ],

        axis=0

    )

    return stacked