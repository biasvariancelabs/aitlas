"""
Notes
-----
    Based on the implementation at:
        https://github.com/CosmiQ/cresi/blob/master/cresi/data_prep/speed_masks.py
"""

import os
import warnings

import cv2
import geopandas as gpd
import math
import numpy as np
import osmnx as osmnx_funcs
import pandas as pd
import skimage.io
from osgeo import gdal, ogr, osr

from aitlas.base import BaseTask
from aitlas.tasks.schemas import SpaceNet5PrepareSpeedMasksTaskSchema


def create_multi_band_geo_tiff(out_path, array):
    """
    Author: Jake Shermeyer
    Array has shape: Channels, Y, X
    """
    driver = gdal.GetDriverByName('GTiff')
    data_set = driver.Create(out_path, array.shape[2], array.shape[1], array.shape[0], gdal.GDT_Byte, ['COMPRESS=LZW'])
    for i, image in enumerate(array, 1):
        data_set.GetRasterBand(i).WriteArray(image)
    del data_set
    return out_path


def gdf_to_array(gdf, im_file, output_raster, burn_value=150, mask_burn_value_key='', compress=True, no_data_value=0,
                 verbose=False):
    """
    Create buffer around geojson for desired geojson feature, save as mask

    Notes
    -----
    https://gis.stackexchange.com/questions/260736/how-to-burn-a-different-value-for-each-polygon-in-a-json-file-using-gdal-rasteri/260737


    Parameters
    ---------
    gdf : gdf
        Input geojson
    im_file : str
        Path to image file corresponding to gdf.
    output_raster : str
        Output path of saved mask (should end in .tif).
    burn_value : int, default 150
        Value to burn to mask. Superseded by mask_burn_value_key.
    mask_burn_value_key : str, default "" in which case burn_value is used
        Column name in gdf to use for mask burning. Supersedes burnValue.
    compress : bool, default True
        Switch to compress output raster.
    no_data_value : int, default 0
        Value to assign array if no data exists. If this value is < 0
        (e.g. -9999), a null value will show in the image.
    verbose : bool, default False
        Switch to print relevant values.

    Returns
    -------
    None
    """
    gdata = gdal.Open(im_file)
    # Set target info
    if compress:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, gdata.RasterXSize, gdata.RasterYSize, 1,
                                                         gdal.GDT_Byte, ['COMPRESS=LZW'])
    else:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster, gdata.RasterXSize, gdata.RasterYSize, 1,
                                                         gdal.GDT_Byte)
    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    if verbose:
        print("gdata.GetGeoTransform():", gdata.GetGeoTransform())
    # Set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    if verbose:
        print("target_ds:", target_ds)
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    out_driver = ogr.GetDriverByName('MEMORY')
    out_data_source = out_driver.CreateDataSource('memData')
    out_layer = out_data_source.CreateLayer("states_extent", raster_srs, geom_type=ogr.wkbMultiPolygon)
    # Burn
    burn_field = "burn"
    id_field = ogr.FieldDefn(burn_field, ogr.OFTInteger)
    out_layer.CreateField(id_field)
    feature_defn = out_layer.GetLayerDefn()
    for j, geom_shape in enumerate(gdf['geometry'].values):
        if verbose:
            print(j, "geom_shape:", geom_shape)
        out_feature = ogr.Feature(feature_defn)
        out_feature.SetGeometry(ogr.CreateGeometryFromWkt(geom_shape.wkt))
        if len(mask_burn_value_key) > 0:
            burn_value = int(gdf[mask_burn_value_key].values[j])
            if verbose:
                print("burn_value:", burn_value)
        else:
            burn_value = burn_value
        out_feature.SetField(burn_field, burn_value)
        out_layer.CreateFeature(out_feature)
    if len(mask_burn_value_key) > 0:
        gdal.RasterizeLayer(target_ds, [1], out_layer,
                            options=["ATTRIBUTE=%s" % burn_field])
    else:
        gdal.RasterizeLayer(target_ds, [1], out_layer, burn_values=[burn_value])


def create_speed_gdf(image_path, geojson_path, mask_path_out_gray, bin_conversion_function,
                     mask_burn_value_key='burnValue', buffer_distance_meters=2, buffer_roundness=1,
                     dissolve_by='inferred_speed_mps', bin_conversion_key='speed_mph', crs=None,
                     zero_fraction_threshold=0.05, verbose=False):
    """
    Create buffer around geojson for speeds, use bin_conversion_func to assign values to the mask.
    """
    try:
        in_gdf = gpd.read_file(geojson_path)
    except:
        print("Can't read geojson:", geojson_path)
        # Create empty mask
        h, w = skimage.io.imread(image_path).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        return []
    if len(in_gdf) == 0:
        print("Empty mask for path:", geojson_path)
        # Create empty mask
        h, w = skimage.io.imread(image_path).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        return []
    # Project
    proj_gdf = osmnx_funcs.project_gdf(in_gdf, to_crs=crs)
    if verbose:
        print("in_gdf.columns:", in_gdf.columns)
    gdf_utm_buffer = proj_gdf.copy()
    # Perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(buffer_distance_meters, buffer_roundness)
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    gdf_buffer = gdf_utm_dissolve.to_crs(in_gdf.crs)
    if verbose:
        print("gdf_buffer['geometry'].values[0]:", gdf_buffer['geometry'].values[0])
    # Set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burn_values = [bin_conversion_function(s) for s in speed_arr]
    gdf_buffer[mask_burn_value_key] = burn_values
    # Create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray, mask_burn_value_key=mask_burn_value_key, verbose=verbose)
    # Check to ensure no mask outside the image pixels (some images are largely black)
    im_bgr = skimage.io.imread(image_path)
    try:
        im_gray = np.sum(im_bgr, axis=2)
        # Check if im_gray is more than X percent black
        zero_fraction = 1. - float(np.count_nonzero(im_gray)) / im_gray.size
        if zero_fraction >= zero_fraction_threshold:
            print("zero_fraction:", zero_fraction)
            print("create_speed_gdf(): checking to ensure masks are null where image is null")
            # Ensure the label doesn't extend beyond the image
            mask_gray = cv2.imread(mask_path_out_gray, 0)
            zero_locations = np.where(im_gray == 0)
            # Set mask_gray to zero at location of zero_locations
            mask_gray[zero_locations] = 0
            # Overwrite
            cv2.imwrite(mask_path_out_gray, mask_gray)
    except:
        # Something is wrong with the image...
        pass
    return gdf_buffer


def convert_array_to_multichannel(in_arr, n_channels=7, burn_value=255,
                                  append_total_band=False, verbose=False):
    """
    Take input array with multiple values, and make each value a unique channel.
    Assume a zero value is background, while value of 1 is the first channel, 2 the second channel, etc.
    """
    h, w = in_arr.shape[:2]
    # Scikit image wants it in this format by default
    out_arr = np.zeros((n_channels, h, w), dtype=np.uint8)
    for band in range(n_channels):
        val = band + 1
        band_out = np.zeros((h, w), dtype=np.uint8)
        if verbose:
            print("band:", band)
        band_arr_bool = np.where(in_arr == val)
        band_out[band_arr_bool] = burn_value
        out_arr[band, :, :] = band_out
    if append_total_band:
        tot_band = np.zeros((h, w), dtype=np.uint8)
        band_arr_bool = np.where(in_arr > 0)
        tot_band[band_arr_bool] = burn_value
        tot_band = tot_band.reshape(1, h, w)
        out_arr = np.concatenate((out_arr, tot_band), axis=0).astype(np.uint8)
    if verbose:
        print("out_arr.shape:", out_arr.shape)
    return out_arr


def speed_masks(geojson_dir, image_dir, output_dir, speed_to_burn_func, mask_burn_val_key='burnValue',
                buffer_distance_meters=2, buffer_roundness=1, dissolve_by='speed_m/s',
                bin_conversion_key='speed_mph', verbose=True, output_dir_multidim='', channel_value_multiplier=1,
                n_channels=8, channel_burn_value=255, append_total_band=True, crs=None):
    """Create speed masks for entire directory."""
    images = sorted([z for z in os.listdir(image_dir) if z.endswith('.tif')])
    for j, image_name in enumerate(images):
        image_root = image_name.split('.')[0]
        image_path = os.path.join(image_dir, image_name)
        mask_path_out = os.path.join(output_dir, image_name)
        geojson_path = os.path.join(geojson_dir, image_root.replace('PS-RGB', 'geojson_roads_speed')
                                    .replace('PS-MS', 'geojson_roads_speed') + '.geojson')
        if (j % 1) == 0:
            print(j + 1, "/", len(images), "image:", image_name, "geojson:", geojson_path)
        if j > 0:
            verbose = False
        create_speed_gdf(image_path, geojson_path, mask_path_out, speed_to_burn_func,
                         mask_burn_value_key=mask_burn_val_key, buffer_distance_meters=buffer_distance_meters,
                         buffer_roundness=buffer_roundness, dissolve_by=dissolve_by,
                         bin_conversion_key=bin_conversion_key, crs=crs, verbose=verbose)
        # If binning
        if output_dir_multidim:
            mask_path_out_md = os.path.join(output_dir_multidim, image_name)
            # Convert array to a multi-channel image
            mask_bins = skimage.io.imread(mask_path_out)
            mask_bins = (mask_bins / channel_value_multiplier).astype(int)
            if verbose:
                print("mask_bins.shape:", mask_bins.shape)
                print("np unique mask_bins:", np.unique(mask_bins))
            # Define mask_channels
            if np.max(mask_bins) == 0:
                h, w = skimage.io.imread(mask_path_out).shape[:2]
                if append_total_band:
                    mask_channels = np.zeros((n_channels + 1, h, w)).astype(np.uint8)
                else:
                    mask_channels = np.zeros((n_channels, h, w)).astype(np.uint8)
            else:
                mask_channels = convert_array_to_multichannel(mask_bins, n_channels=n_channels,
                                                              burn_value=channel_burn_value,
                                                              append_total_band=append_total_band, verbose=verbose)
            if verbose:
                print("mask_channels.shape:", mask_channels.shape)
                print("mask_channels.dtype:", mask_channels.dtype)
            # Write to file
            create_multi_band_geo_tiff(mask_path_out_md, mask_channels)


class SpaceNet5PrepareSpeedMasksTask(BaseTask):
    """
    Implements the functionality of step 03 in the CRESI framework.
    Creates the segmented speed masks from the processed images.
    There are two types of masks being created:
        1. Continuous - One-channel mask where the value of the pixel is proportional to the speed of the road
        2. Multi-channel (or binned) masks where each channel corresponds to a speed range (10-20mph, 20-30mph, etc.)
    """
    schema = SpaceNet5PrepareSpeedMasksTaskSchema  # set up the task schema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
            config : Config
        """
        super().__init__(model, config)

    def run(self):
        """Implements the main logic of the task."""
        buffer_roundness = 1
        mask_burn_val_key = 'burnValue'
        dissolve_by = 'inferred_speed_mps'
        bin_conversion_key = 'inferred_speed_mph'
        verbose = True
        # Skimage throws an annoying "low contrast warning, so ignore"
        warnings.filterwarnings("ignore")
        # Continuous case, skips converting to multi-channel
        if len(self.config.output_mask_multidim_dir) == 0:
            min_road_burn_val = 0
            min_speed_continuous = 0
            max_speed_continuous = 65
            mask_max = 255
            verbose = True
            # Placeholder variables for binned case
            channel_value_multiplier, n_channels, channel_burn_value, append_total_band = 0, 0, 0, 0
            # Make output dir
            os.makedirs(self.config.output_mask_dir_contin, exist_ok=True)

            def speed_to_burn_func(speed):
                """Convert speed estimate to mask burn value between 0 and mask_max"""
                bw = mask_max - min_road_burn_val
                burn_val = min(min_road_burn_val + bw * (
                        (speed - min_speed_continuous) / (max_speed_continuous - min_speed_continuous)), mask_max)
                return max(burn_val, min_road_burn_val)

            speed_arr_continuous = np.arange(min_speed_continuous, max_speed_continuous + 1, 1)
            burn_val_arr = [speed_to_burn_func(s) for s in speed_arr_continuous]
            d = {'burn_val': burn_val_arr, 'speed': speed_arr_continuous}
            df_s = pd.DataFrame(d)
            # Make conversion dataframe (optional)
            if not os.path.exists(self.config.output_conversion_csv_contin):
                print("Write burn_val -> speed conversion to:", self.config.output_conversion_csv_contin)
                df_s.to_csv(self.config.output_conversion_csv_contin)
            else:
                print("Path already exists, not overwriting...", self.config.output_conversion_csv_contin)
        # Multi-channel case
        else:
            min_speed_bin = 1
            max_speed_bin = 65
            channel_burn_value = 255
            channel_value_multiplier = 1
            append_total_band = True
            speed_arr_bin = np.arange(min_speed_bin, max_speed_bin + 1, 1)
            # Make output dir
            if len(self.config.output_mask_dir_contin) > 0:
                os.makedirs(self.config.output_mask_dir_contin, exist_ok=True)
            if len(self.config.output_mask_multidim_dir) > 0:
                os.makedirs(self.config.output_mask_multidim_dir, exist_ok=True)
            bin_size_mph = 10.0

            def speed_to_burn_func(speed_mph):
                """
                Bin every 10 mph or so.
                Convert speed estimate to appropriate channel bin = 0 if speed = 0.
                """
                return int(int(math.ceil(speed_mph / bin_size_mph)) * channel_value_multiplier)

            n_channels = len(np.unique([int(speed_to_burn_func(z)) for z in speed_arr_bin]))
            print("n_channels:", n_channels)
            channel_value_multiplier = int(255 / n_channels)
            # Make conversion dataframe
            print("speed_arr_bin:", speed_arr_bin)
            burn_val_arr = np.array([speed_to_burn_func(s) for s in speed_arr_bin])
            print("burn_val_arr:", burn_val_arr)
            d = {'burn_val': burn_val_arr, 'speed': speed_arr_bin}
            df_s_bin = pd.DataFrame(d)
            # Add a couple columns, first the channel that the speed corresponds to
            channel_val = (burn_val_arr / channel_value_multiplier).astype(int) - 1
            print("channel_val:", channel_val)
            df_s_bin['channel'] = channel_val
            # Make conversion dataframe (optional)
            if not os.path.exists(self.config.output_conversion_csv_binned):
                print("Write burn_val -> speed conversion to:", self.config.output_conversion_csv_binned)
                df_s_bin.to_csv(self.config.output_conversion_csv_binned)
            else:
                print("path already exists, not overwriting...", self.config.output_conversion_csv_binned)
        speed_masks(self.config.geojson_dir, self.config.image_dir, self.config.output_mask_dir_contin,
                    speed_to_burn_func, mask_burn_val_key=mask_burn_val_key,
                    buffer_distance_meters=self.config.buffer_distance_meters, buffer_roundness=buffer_roundness,
                    dissolve_by=dissolve_by, bin_conversion_key=bin_conversion_key, verbose=verbose,
                    output_dir_multidim=self.config.output_mask_multidim_dir,
                    channel_value_multiplier=channel_value_multiplier, n_channels=n_channels,
                    channel_burn_value=channel_burn_value, append_total_band=append_total_band)
