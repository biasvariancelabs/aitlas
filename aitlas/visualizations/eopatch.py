import matplotlib.pyplot as plt
import pandas as pd

import os

from eolearn.core import EOPatch, FeatureType
from eolearn.geometry import VectorToRasterTask

def display_eopatch_predictions(eopatches_path, patch, y_pred, test_index, output_path):
    eop = EOPatch.load(eopatches_path+os.sep+patch)
    polygons = eop.vector_timeless["CROP_TYPE_GDF"]
    predictions_list = []
    for row in polygons.itertuples():
        current_path = patch+os.sep+str(int(row.polygon_id))
        if current_path in test_index.path.values:
            label = y_pred[test_index.index[test_index['path'] == current_path].values[0]]
            print(int(label))
            predictions_list.append(int(label)+1) # temporary, not consistent with classmapping
        else:
            print("Not found")
            predictions_list.append(0) # essentially background as in VectorToRasterTask, should be changed
            #len(list(test_dataset.mapping.index.values)))


    pred_polygons = polygons.copy()
    pred_polygons['ct_pred'] = pd.Series(data=predictions_list, index=pred_polygons.index)

    # keep this if we need the true mask as well at some point
    # pred_polygons['ct_true'] = pd.Series(data=[test_dataset.mapping.loc[x].id if x in test_dataset.mapping.index.values else -1 for x in pred_polygons.ct_eu_code], index=pred_polygons.index)

    temp = VectorToRasterTask(vector_input=pred_polygons,
    raster_feature=(FeatureType.MASK_TIMELESS, 'poly'),
    values_column='ct_pred',
    raster_shape = (FeatureType.MASK_TIMELESS, 'CROP_TYPE')
    )
    croptype_indicator_mask = temp.execute(eop).mask_timeless['poly']

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    im = ax.imshow(croptype_indicator_mask)
    fig.colorbar(im)
    fig.savefig(output_path+os.sep+"eop"+str(patch)+"_visual_predictions.png", dpi=300)

