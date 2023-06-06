"""Contains classes for image transformations specific for BreizhCrops dataset."""

import torch
import numpy as np

from ..base import BaseTransforms

BANDS = {
    "L1C": [
        "B1",
        "B10",
        "B11",
        "B12",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "QA10",
        "QA20",
        "QA60",
        "doa",
        "label",
        "id",
    ],
    "L2A": [
        "doa",
        "id",
        "code_cultu",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "CLD",
        "EDG",
        "SAT",
    ],
}

SELECTED_BANDS = {
    "L1C": [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B10",
        "B11",
        "B12",
        "QA10",
        "QA20",
        "QA60",
        "doa",
    ],
    "L2A": [
        "doa",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "CLD",
        "EDG",
        "SAT",
    ],
}


class SelectBands(BaseTransforms):
    """
    A class used to select and process spectral bands from satellite data.


    :param level: satellite data level to be processed ("L1C" or "L2A")
    :type level: str

    .. note::
        This class requires a level argument at initialization.
        This should be one of the predefined satellite data levels ("L1C" or "L2A").
    """

    configurables = ["level"]

    def __init__(self, *args, **kwargs):
        """
        Initialize the SelectBands class by setting the satellite data level.

        """
        BaseTransforms.__init__(self, *args, **kwargs)

        self.level = kwargs["level"]

        # padded_value = PADDING_VALUE
        self.sequencelength = 45

        bands = BANDS[self.level]
        if self.level == "L1C":
            selected_bands = [
                "B1",
                "B10",
                "B11",
                "B12",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B9",
            ]
        elif self.level == "L2A":
            selected_bands = [
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B8A",
                "B11",
                "B12",
            ]

        self.selected_band_idxs = np.array([bands.index(b) for b in selected_bands])

    def __call__(self, input, target=None):
        """
        Process the input and target data, apply transformation and return the result.
        Transformation includes selecting bands, scaling, and replacing short seqences (if necessary).

        :param input: input data to be processed
        :type input: tuple
        :param target: target data, defaults to None
        :type target: tensor, optional
        :return: processed input and target data
        :rtype: tuple
        """
        # x = x[x[:, 0] != padded_value, :]  # remove padded values

        # choose selected bands
        x, y = input
        x = x[:, self.selected_band_idxs] * 1e-4  # scale reflectances to 0-1

        # choose with replacement if sequencelength smaller als choose_t
        replace = False if x.shape[0] >= self.sequencelength else True
        idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
        idxs.sort()

        x = x[idxs]

        return torch.from_numpy(x).type(torch.FloatTensor), torch.tensor(
            y, dtype=torch.long
        )
