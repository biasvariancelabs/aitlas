import os
from textwrap import wrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import VectorToRasterTask
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize


def display_eopatch_predictions(
    eopatches_path, patch, y_pred, test_index, y_true, classmapping
):
    eop = EOPatch.load(eopatches_path + os.sep + patch)
    polygons = eop.vector_timeless["CROP_TYPE_GDF"]
    predictions_list = []
    true_list = []
    for row in polygons.itertuples():
        current_path = patch + os.sep + str(int(row.polygon_id))
        if current_path in test_index.path.values:
            label = y_pred[
                test_index.index[test_index["path"] == current_path].values[0]
            ]
            true_label = y_true[
                test_index.index[test_index["path"] == current_path].values[0]
            ]
            predictions_list.append(
                int(label)
            )  # temporary, not consistent with classmapping
            true_list.append(int(true_label))
        else:
            predictions_list.append(
                11
            )  # essentially background as in VectorToRasterTask, should be changed
            true_list.append(11)

    pred_polygons = polygons.copy()
    pred_polygons["ct_pred"] = pd.Series(
        data=predictions_list, index=pred_polygons.index
    )

    n_classes = len(classmapping.index)

    temp = VectorToRasterTask(
        vector_input=pred_polygons,
        raster_feature=(FeatureType.MASK_TIMELESS, "poly"),
        values_column="ct_pred",
        raster_shape=(FeatureType.MASK_TIMELESS, "CROP_TYPE"),
        no_data_value=11,
    )
    croptype_indicator_mask = temp.execute(eop).mask_timeless["poly"]
    values = np.unique(croptype_indicator_mask.ravel())

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot()

    croptype_indicator_mask = croptype_indicator_mask
    tab10_map = cm.get_cmap("tab10", n_classes)  # If it's not 10 this wont work?
    colors = [tuple(col) for col in tab10_map.colors] + [
        (1, 1, 1)
    ]  # this or list all colors from mapping csv (not implemented yet)

    cmap = LinearSegmentedColormap.from_list("mycmap", colors, N=n_classes + 1)

    im = ax.imshow(
        croptype_indicator_mask,
        norm=Normalize(vmin=0, vmax=len(classmapping.index.values)),
        cmap=cmap,
    )  #'nipy_spectral') #gist_ncar

    # get the colors of the values, according to the
    # colormap used by imshow
    colors = [im.cmap(im.norm(value)) for value in values]

    # create a patch (proxy artist) for every color

    labels = [
        classmapping[classmapping.id == cl_id].classname.values[0]
        for cl_id in values[:-1]
    ] + ["Background"]
    labels = ["\n".join(wrap(l, 22)) for l in labels]
    patches = [
        mpatches.Patch(edgecolor=(0, 0, 0), facecolor=colors[i], label=labels[i])
        for i in range(len(values))
    ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.subplots_adjust(left=0.15, right=0.7)

    return fig
