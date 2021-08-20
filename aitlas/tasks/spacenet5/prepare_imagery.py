"""
Notes
-----
    Based on the implementation at:
        https://github.com/CosmiQ/cresi/blob/master/cresi/data_prep/create_8bit_images.py
    Tutorial for installing GDAL on Linux systems:
        https://ljvmiranda921.github.io/notebook/2019/04/13/install-gdal/#using-your-package-manager
"""
import os
from multiprocessing.pool import Pool

import numpy as np
from osgeo import gdal

from ...base import BaseTask
from ..schemas import SpaceNet5PrepareImageryTaskSchema


rescale = {
    "tot_3band": {1: [63, 1178], 2: [158, 1285], 3: [148, 880]},
    # RGB corresponds to bands: 5, 3, 2
    "tot_8band": {
        1: [154, 669],
        2: [122, 1061],
        3: [119, 1520],
        4: [62, 1497],
        5: [20, 1342],
        6: [36, 1505],
        7: [17, 1853],
        8: [7, 1559],
    },
}


def convert_to_8bit(
    input_raster,
    output_raster,
    output_pix_type="Byte",
    output_format="GTiff",
    rescale_type="perc",
    percentiles=None,
    band_order=None,
    no_data_value=0,
    max_zero_fraction=0.3,
):
    """
    Convert 16 bit image to 8 bit.

    Parameters
    ----------
        input_raster
        output_raster
        output_pix_type
        output_format
        rescale_type : str, one of  ("clip", "perc", "dict")
            If "clip", scaling is done strictly between [0, 65535].
            if "perc", each band is rescaled to a min and max set by percentiles.
            if "dict", access the 'rescale' dict at the beginning for rescaling percentiles.
        percentiles : list
        band_order : list
            Determines which bands and in what order to create them.
            If it is empty, is uses all bands. For WV3 8-band, RGB corresponds to 5, 3, 2:
                https://gdal.org/programs/gdal_translate.html
        no_data_value
        max_zero_fraction : float
            Ff the images has greater than max_zero_fraction == 0, then it is skipped.
    """
    if band_order is None:
        band_order = []
    if percentiles is None:
        percentiles = [2, 98]
    src_raster = gdal.Open(input_raster)
    if len(band_order) == 0:
        n_bands = src_raster.RasterCount
    else:
        n_bands = len(band_order)
    if n_bands == 3:
        cmd = [
            "gdal_translate",
            "-ot",
            output_pix_type,
            "-of",
            output_format,
            "-a_nodata",
            str(no_data_value),
            "-co",
            '"PHOTOMETRIC=rgb"',
        ]
    else:
        cmd = [
            "gdal_translate",
            "-ot",
            output_pix_type,
            "-of",
            output_format,
            "-a_nodata",
            str(no_data_value),
        ]
    # Get bands
    if len(band_order) == 0:
        band_list = range(1, src_raster.RasterCount + 1)
    else:
        band_list = band_order
    # Iterate through bands
    for j, bandId in enumerate(band_list):
        band = src_raster.GetRasterBand(bandId)
        if rescale_type == "perc":
            b_min = band.GetMinimum()
            b_max = band.GetMaximum()
            band_arr_tmp = band.ReadAsArray()
            band_arr_flat = band_arr_tmp.flatten()
            if b_min is None or b_max is None:
                b_min, b_max = np.min(band_arr_flat), np.max(band_arr_flat)
            print("b_min, b_max:", b_min, b_max)
            band_arr_pos = band_arr_flat[band_arr_flat > 0]
            # Test zero fraction
            zero_fraction = 1.0 - (len(band_arr_pos) / (1.0 * len(band_arr_flat)))
            print("zero_fraction = ", zero_fraction)
            if zero_fraction >= max_zero_fraction:
                cmd_str = "echo " + input_raster + " too many zeros, skpping"
                print("zero_fraction = ", zero_fraction, "skipping...")
                return cmd_str
            if len(band_arr_pos) == 0:
                (b_min, b_max) = band.ComputeRasterMinMax(1)
            else:
                b_min = np.percentile(band_arr_pos, percentiles[0])
                b_max = np.percentile(band_arr_pos, percentiles[1])
        elif rescale_type == "clip":
            b_min, b_max = 0, 65535
        else:
            b_min, b_max = rescale[rescale_type][bandId]
        b_min = max(1, b_min)
        cmd.append("-b {}".format(bandId))
        cmd.append("-scale_{}".format(j + 1))
        cmd.append("{}".format(b_min))
        cmd.append("{}".format(b_max))
        cmd.append("{}".format(0))
        cmd.append("{}".format(255))
    cmd.append(input_raster)
    cmd.append(output_raster)
    cmd_str = " ".join(cmd)
    print("Conversion_command list:", cmd)
    print("Conversion_command str:", cmd_str)
    try:
        os.system(cmd_str)
        return cmd_str
    except:
        return cmd_str


def prepare_image(params):
    (
        im_file,
        im_file_raw,
        im_file_out,
        output_pix_type,
        output_format,
        rescale_type,
        percentiles,
        max_zero_fraction,
        band_order,
    ) = params
    if not im_file.endswith(".tif"):
        return
    if not os.path.isfile(im_file_out):
        convert_to_8bit(
            im_file_raw,
            im_file_out,
            output_pix_type=output_pix_type,
            output_format=output_format,
            rescale_type=rescale_type,
            percentiles=percentiles,
            band_order=band_order,
            max_zero_fraction=max_zero_fraction,
        )
    else:
        print("File exists, skipping!", im_file_out)


class SpaceNet5PrepareImageryTask(BaseTask):
    """
    Implements the functionality of step 02 in the CRESI framework.
    Extracts the 8-bit RGB imagery from the 16-bit multi-spectral (pan-sharped) imagery.
    """

    schema = SpaceNet5PrepareImageryTaskSchema  # set up the task schema

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
        # Parse band order
        if len(self.config.band_order) == 0:
            band_order = []
        else:
            band_order_str = self.config.band_order.split(",")
            band_order = [int(z) for z in band_order_str]
        # Parse percentiles
        percentiles = [int(z) for z in self.config.percentiles.split(",")]
        # Values that should remain constant
        output_pix_type = "Byte"
        output_format = "GTiff"
        max_zero_fraction = 0.3
        path_images_raw = self.config.in_dir
        path_images_8bit = self.config.out_dir
        os.makedirs(path_images_8bit, exist_ok=True)
        # Iterate through images and them convert to 8-bit
        im_files = [
            z for z in sorted(os.listdir(path_images_raw)) if z.endswith(".tif")
        ]
        print("im_files:", im_files)
        params = []
        for i, im_file in enumerate(im_files):
            # Create an 8-bit image
            im_file_raw = os.path.join(path_images_raw, im_file)
            im_file_out = os.path.join(path_images_8bit, im_file)
            params.append(
                (
                    im_file,
                    im_file_raw,
                    im_file_out,
                    output_pix_type,
                    output_format,
                    self.config.rescale_type,
                    percentiles,
                    max_zero_fraction,
                    band_order,
                )
            )
        pool = Pool(self.config.num_threads)
        pool.map(prepare_image, params)
