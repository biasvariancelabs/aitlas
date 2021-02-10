"""
Notes
-----
    Based on the implementation at:
        https://github.com/CosmiQ/cresi/blob/master/cresi/06_infer_speed.py
"""
import logging
import os
import time
from multiprocessing.pool import Pool

import cv2
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import scipy.spatial
import shapely
import skimage.io
from matplotlib import collections as mpl_collections
from matplotlib import pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW

from aitlas.base import BaseTask
from .schemas import SpaceNet5InferSpeedTaskSchema

# Create or get the logger
logger = logging.getLogger(__name__)
# Set log level
logger.setLevel(logging.INFO)


def plot_graph_on_im_yuge(g_, im_test_file, fig_size=(12, 12), show_end_nodes=False, width_key='inferred_speed_mps',
                          width_multiplier=0.125, default_node_size=15, node_color='#0086CC', edge_color='#00a6ff',
                          node_edge_color='none', title='', fig_name='', max_speeds_per_line=12, line_alpha=0.5,
                          node_alpha=0.6, default_dpi=300, plt_save_quality=75, ax=None, verbose=True,
                          super_verbose=False):
    """
    Copied verbatim from apls_tools.py

    Overlay graph on image, if width_key == int, use a constant width
    """
    # Set dpi to approximate native resolution
    # mpl can handle a max of 2^29 pixels, or 23170 on a side
    # recompute max_dpi
    max_dpi = int(23000 / max(fig_size))
    try:
        im_cv2 = cv2.imread(im_test_file, 1)
        img_mpl = cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB)
    except:
        img_sk = skimage.io.imread(im_test_file)
        # Make sure image is h, w, channels (assume less than 20 channels)
        if (len(img_sk.shape) == 3) and (img_sk.shape[0] < 20):
            img_mpl = np.moveaxis(img_sk, 0, -1)
        else:
            img_mpl = img_sk
    h, w = img_mpl.shape[:2]
    # Make fig_size proportional to shape
    if h > w:
        fig_size = (fig_size[1] * 1. * w / h, fig_size[1])
    elif w > h:
        fig_size = (fig_size[0], fig_size[0] * 1. * h / w)
    else:
        pass
    if h > 10000 or w > 10000:
        fig_size = (2 * fig_size[0], 2 * fig_size[1])
        max_dpi = int(23000 / max(fig_size))
    if verbose:
        print("img_mpl.shape: " + str(img_mpl.shape))
    desired_dpi = max(default_dpi, int(np.max(img_mpl.shape) / np.max(fig_size)))
    if verbose:
        print("desired dpi: " + str(desired_dpi))
    # Max out dpi at 3500
    dpi = int(np.min([max_dpi, desired_dpi]))
    if verbose:
        print("figsize: " + str(fig_size))
        print("plot dpi: " + str(dpi))
    node_x, node_y, lines, widths, title_values = [], [], [], [], []
    x_set, y_set = set(), set()
    # Get edge data
    for i, (u, v, edge_data) in enumerate(g_.edges(data=True)):
        # If type(edge_data['geometry_pix'])
        coordinates = list(edge_data['geometry_pix'].coords)
        if super_verbose:
            print("\n" + str(i) + " " + str(u) + " " + str(v) + " " + str(edge_data))
            print("edge_data: " + str(edge_data))
            print("  coords: " + str(coordinates))
        lines.append(coordinates)
        # Point 0
        xp = coordinates[0][0]
        yp = coordinates[0][1]
        if not ((xp in x_set) and (yp in y_set)):
            node_x.append(xp)
            x_set.add(xp)
            node_y.append(yp)
            y_set.add(yp)
        # Point 1
        xp = coordinates[-1][0]
        yp = coordinates[-1][1]
        if not ((xp in x_set) and (yp in y_set)):
            node_x.append(xp)
            x_set.add(xp)
            node_y.append(yp)
            y_set.add(yp)
        if type(width_key) == str:
            if super_verbose:
                print("edge_data[width_key]: " + str(edge_data[width_key]))
            width = int(np.rint(edge_data[width_key] * width_multiplier))
            title_values.append(int(np.rint(edge_data[width_key])))
        else:
            width = width_key
        widths.append(width)
    # Get nodes
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.imshow(img_mpl)
    # Plot segments, scale widths with dpi
    widths = (default_dpi / float(dpi)) * np.array(widths)
    lc = mpl_collections.LineCollection(lines, colors=edge_color, linewidths=widths, alpha=line_alpha)
    ax.add_collection(lc)
    # Plot nodes?
    if show_end_nodes:
        # Scale size with dpi
        node_size = max(0.01, (default_node_size * default_dpi / float(dpi)))
        # Node_size = 3
        if verbose:
            print("node_size: " + str(node_size))
        ax.scatter(node_x, node_y, c=node_color, s=node_size, alpha=node_alpha, edgecolor=node_edge_color, zorder=1)
    ax.axis('off')
    # Title
    if len(title_values) > 0:
        if verbose:
            print("title_vals: " + str(title_values))
        title_strings = np.sort(np.unique(title_values)).astype(str)
        # Split title str if it's too long
        if len(title_strings) > max_speeds_per_line:
            # Construct new title string
            n, b = max_speeds_per_line, title_strings
            title_strings = np.insert(b, range(n, len(b), n), "\n")
        if verbose:
            print("title_strs: " + str(title_strings))
        title_new = title + '\n' + width_key + " = " + " ".join(title_strings)
    else:
        title_new = title
    if title:
        ax.set_title(title_new)
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.96)
    if fig_name:
        print("Saving to: " + str(fig_name))
        if dpi > 1000:
            plt_save_quality = 50
        plt.savefig(fig_name, dpi=dpi, quality=plt_save_quality)
    return ax


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    Parameters
    ----------
        values : Numpy ndarray
        weights : Numpy ndarray with the same shape
    """
    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)
    mean = weighted_stats.mean  # weighted mean of data (equivalent to np.average(array, weights=weights))
    std = weighted_stats.std  # standard deviation with default degrees of freedom correction
    var = weighted_stats.var  # variance with default degrees of freedom correction
    return mean, std, var


def load_speed_conversion_dict_continuous(csv_location):
    """
    Load speed to burn_val conversion dataframe and create conversion dictionary.
    Assume continuous conversion
    """
    df_ = pd.read_csv(csv_location, index_col=0)
    # Get dict of pixel value to speed
    df_tmp = df_.set_index('burn_val')
    dic = df_tmp.to_dict()['speed']
    return df_, dic


def get_nearest_key(dic, value):
    """Get nearest dict key to the input value."""
    my_list = dic
    key = min(my_list, key=lambda x: abs(x - value))
    return key


def load_speed_conversion_dict_binned(csv_location, speed_increment=5):
    """
    Load speed to burn_val conversion dataframe and create conversion dictionary.
    speed_increment is the increment of speed limits in mph 10 mph bins go from 1-10, and 21-30, etc.
    Breakdown of speed limits in training set:
        15.0 5143
        18.75 6232
        20.0 18098
        22.5 347
        25.0 16526
        26.25 50
        30.0 734
        33.75 26
        35.0 3583
        41.25 16
        45.0 2991
        48.75 17
        55.0 2081
        65.0 407
    """
    df_ = pd.read_csv(csv_location, index_col=0)
    # Get dict of channel to speed
    df = df_[['channel', 'speed']]
    # Simple mean of speed bins
    means = df.groupby(['channel']).mean().astype(int)
    dic = means.to_dict()['speed']
    # Speeds are every 5 mph, so take the mean of the 5 mph bins
    # or just add increment/2 to means...
    dic.update((x, y + speed_increment / 2) for x, y in dic.items())
    # OPTIONAL
    # If using 10mph bins, update dic
    dic[0] = 7.5
    dic[1] = 17.5  # 15, 18.75, and 20 are all common
    dic[2] = 25  # 25 mph speed limit is ubiquitous
    dic[3] = 35  # 35 mph speed limit is ubiquitous
    dic[4] = 45  # 45 mph speed limit is ubiquitous
    dic[5] = 55  # 55 mph speed limit is ubiquitous
    dic[6] = 65  # 65 mph speed limit is ubiquitous
    return df_, dic


def get_linestring_midpoints(geom):
    """
    Get midpoints of each line segment in the line.
    Also return the length of each segment, assuming cartesian coordinates.
    """
    coordinates = list(geom.coords)
    N = len(coordinates)
    x_middles, y_middles, dls = [], [], []
    for i in range(N - 1):
        (x0, y0) = coordinates[i]
        (x1, y1) = coordinates[i + 1]
        x_middles.append(np.rint(0.5 * (x0 + x1)))
        y_middles.append(np.rint(0.5 * (y0 + y1)))
        dl = scipy.spatial.distance.euclidean(coordinates[i], coordinates[i + 1])
        dls.append(dl)
    return np.array(x_middles).astype(int), np.array(y_middles).astype(int), np.array(dls)


def get_patch_speed_single_channel(patch, conv_dict, percentile=80, verbose=False):
    """
    Get the estimated speed of the given patch where the value of the 2-D mask translates directly to speed.
    """
    # Get mean of all high values
    thresh = np.percentile(patch, percentile)
    indices = np.where(patch >= thresh)
    patch_filter = patch[indices]
    # Get mean of high percentiles
    pixel_val = np.median(patch_filter)
    # Get nearest key to pixel_val
    key = get_nearest_key(conv_dict, pixel_val)
    speed = conv_dict[key]
    if verbose:
        logger.info("patch_filter: " + str(patch_filter))
        logger.info("conv_dict: " + str(conv_dict))
        logger.info("key: " + str(key))
        logger.info("speed: " + str(speed))
    return speed, patch_filter


def get_patch_speed_multichannel(patch, conv_dict, min_z=128, weighted=True, percentile=90, verbose=False,
                                 super_verbose=False):
    """
    Get the estimated speed of the given patch where each channel corresponds to a different speed bin.
    Assume patch has shape: (channels, h, w).
    If weighted, take weighted mean of each band above threshold, else assign speed to max band.
    """
    # Set minimum speed if no channel his min_z
    min_speed = -1
    # Could use mean, max, or percentile
    z_val_vec = np.rint(np.percentile(patch, percentile, axis=(1, 2)).astype(int))
    if verbose:
        logger.info("    z_val_vec: " + str(z_val_vec))
    if not weighted:
        best_idx = np.argmax(z_val_vec)
        if z_val_vec[best_idx] >= min_z:
            speed_out = conv_dict[best_idx]
        else:
            speed_out = min_speed
    else:
        # Take a weighted average of all bands with all values above the threshold
        speeds, weights = [], []
        for band, speed in conv_dict.items():
            if super_verbose:
                logger.info("    band: " + str(band), "speed;", str(speed))
            if z_val_vec[band] > min_z:
                speeds.append(speed)
                weights.append(z_val_vec[band])
                # Get mean speed
        if len(speeds) == 0:
            speed_out = min_speed
        # Get weighted speed
        else:
            speed_out, std, var = weighted_avg_and_std(speeds, weights)
            if verbose:
                logger.info("    speeds: " + str(speeds), "weights: " + str(weights))
                logger.info("    w_mean: " + str(speed_out), "std: " + str(std))
            if (type(speed_out) == list) or (type(speed_out) == np.ndarray):
                speed_out = speed_out[0]
    if verbose:
        logger.info("    speed_out: " + str(speed_out))
    return speed_out, z_val_vec


def get_edge_time_properties(mask, edge_data, conv_dict, min_z=128, dx=4, dy=4, percentile=80, max_speed_band=-2,
                             use_weighted_mean=True, variable_edge_speed=False, verbose=True):
    """
    Get speed estimate from proposal mask and graph edge_data by inferring the speed along each segment
    based on the coordinates in the output mask.

    min_z is the minimum mask value to consider a hit for speed
    dx, dy is the patch size to average for speed
    if totband, the final band of the mask is assumed to just be a binary road mask and not correspond to a speed bin
    if weighted_mean, sum up? the weighted mean of speeds in the multichannel case
    """
    meters_to_miles = 0.000621371
    if len(mask.shape) > 2:
        multichannel = True
    else:
        multichannel = False
    # Get coordinates
    if verbose:
        logger.info("edge_data: " + str(edge_data))
    length_pix = np.sum([edge_data['length_pix']])
    length_m = edge_data['length']
    pix_to_meters = length_m / length_pix
    length_miles = meters_to_miles * length_m
    if verbose:
        logger.info("length_pix: " + str(length_pix))
        logger.info("length_m: " + str(length_m))
        logger.info("length_miles: " + str(length_miles))
        logger.info("pix_to_meters: " + str(pix_to_meters))
    wkt_pix = edge_data['wkt_pix']
    geom_pix = edge_data['geometry_pix']
    if type(geom_pix) == str:
        geom_pix = shapely.wkt.loads(wkt_pix)
    # Get points
    coordinates = list(geom_pix.coords)
    if verbose:
        logger.info("type geom_pix: " + str(type(geom_pix)))
        logger.info("wkt_pix: " + str(wkt_pix))
        logger.info("geom_pix: " + str(geom_pix))
        logger.info("coords: " + str(coordinates))
    # Get midpoints of each segment in the linestring
    x_mids, y_mids, dls = get_linestring_midpoints(geom_pix)
    if verbose:
        logger.info("x_mids: " + str(x_mids))
        logger.info("y_mids: " + str(y_mids))
        logger.info("dls: " + str(dls))
        logger.info("np.sum dls (pix): " + str(np.sum(dls)))
        logger.info("edge_data.length (m): " + str(edge_data['length']))
    # for each midpoint:
    #   1. access that portion of the mask, +/- desired pixels
    #   2. get speed and travel time
    #   Sum the travel time for each segment to get the total speed, this
    #   means that the speed is variable along the edge
    # Could also sample the mask at each point in the linestring,
    # except endpoints which would give a denser estimate of speed
    total_hours = 0
    speed_array = []
    z_arr = []
    for j, (x, y, dl_pix) in enumerate(zip(x_mids, y_mids, dls)):
        x0, x1 = max(0, x - dx), x + dx + 1
        y0, y1 = max(0, y - dy), y + dy + 1
        if verbose:
            logger.info("  x, y, dl: " + str(x), str(y), str(dl_pix))
        # Multichannel case...
        if multichannel:
            patch = mask[:, y0:y1, x0:x1]
            n_channels, h, w = mask.shape
            if max_speed_band < n_channels - 1:
                patch = patch[:max_speed_band + 1, :, :]
                # # Assume the final channel is total, so cut it out
            if verbose:
                logger.info("  patch.shape: " + str(patch.shape))
            # Get estimated speed of mask patch
            speed_mph_seg, z = get_patch_speed_multichannel(patch, conv_dict, percentile=percentile, min_z=min_z,
                                                            weighted=use_weighted_mean, verbose=verbose)
        else:
            patch = mask[y0:y1, x0:x1]
            z = 0
            speed_mph_seg, _ = get_patch_speed_single_channel(patch, conv_dict, percentile=percentile, verbose=verbose)
        # Add to arrays
        speed_array.append(speed_mph_seg)
        z_arr.append(z)
        length_m_seg = dl_pix * pix_to_meters
        length_miles_seg = meters_to_miles * length_m_seg
        hours = length_miles_seg / speed_mph_seg
        total_hours += hours
        if verbose:
            logger.info("  speed_mph_seg: " + str(speed_mph_seg))
            logger.info("  dl_pix: " + str(dl_pix), "length_m_seg", str(length_m_seg),
                        "length_miles_seg: " + str(length_miles_seg))
            logger.info("  hours: " + str(hours))
    # Get edge properties
    if variable_edge_speed:
        mean_speed_mph = length_miles / total_hours
    else:
        # Assume that the edge has a constant speed, so guess the total speed
        if multichannel:
            # Get most common channel, assign that channel as mean speed
            z_arr = np.array(z_arr)
            # Sum along the channels
            z_vec = np.sum(z_arr, axis=0)
            # Get max speed value
            channel_best = np.argmax(z_vec)
            if verbose:
                logger.info("z_arr: " + str(z_arr))
                logger.info("z_vec: " + str(z_vec))
                logger.info("channel_best: " + str(channel_best))
            mean_speed_mph = conv_dict[channel_best]
            # Reassign total hours
            total_hours = length_miles / mean_speed_mph
        else:
            # Or always use variable edge speed?
            mean_speed_mph = length_miles / total_hours
    if verbose:
        logger.info("tot_hours: " + str(total_hours))
        logger.info("mean_speed_mph: " + str(mean_speed_mph))
        logger.info("length_miles: " + str(length_miles))
    return total_hours, mean_speed_mph, length_miles


def infer_travel_time(params):
    """
    Get an estimate of the average speed and travel time of each edge
    in the graph from the mask and conversion dictionary
    For each edge, get the geometry in pixel coordinates
      For each point, get the nearest neighbors in the maks and infer
      the local speed
    """
    g_, mask, conv_dict, min_z, dx, dy, percentile, max_speed_band, use_weighted_mean, variable_edge_speed, verbose, \
    out_file, save_geo_packages, im_root, graph_dir_out = params
    print("im_root:", im_root)
    mph_to_mps = 0.44704  # miles per hour to meters per second
    pickle_protocol = 4
    for i, (u, v, edge_data) in enumerate(g_.edges(data=True)):
        if verbose:
            logger.info("\n" + str(i) + " " + str(u) + " " + str(v) + " " + str(edge_data))
        if (i % 1000) == 0:
            logger.info(str(i) + " / " + str(len(g_.edges())) + " edges")
        tot_hours, mean_speed_mph, length_miles = get_edge_time_properties(mask, edge_data, conv_dict, min_z=min_z,
                                                                           dx=dx, dy=dy, percentile=percentile,
                                                                           max_speed_band=max_speed_band,
                                                                           use_weighted_mean=use_weighted_mean,
                                                                           variable_edge_speed=variable_edge_speed,
                                                                           verbose=verbose)
        # Update edges
        edge_data['Travel Time (h)'] = tot_hours
        edge_data['inferred_speed_mph'] = np.round(mean_speed_mph, 2)
        edge_data['length_miles'] = length_miles
        edge_data['inferred_speed_mps'] = np.round(mean_speed_mph * mph_to_mps, 2)
        edge_data['travel_time_s'] = np.round(3600. * tot_hours, 3)
    g = g_.to_undirected()
    # Save graph
    nx.write_gpickle(g, out_file, protocol=pickle_protocol)
    # Save GeoPackage as well?
    if save_geo_packages:
        g_out = g
        logger.info("Saving geoPackage to directory: {}".format(graph_dir_out))
        filename = im_root + '.gpkg'  # append GeoPackage extension to the filename so it doesn't raise warnings
        filepath = os.path.join(graph_dir_out, filename)
        # https://github.com/gboeing/osmnx/issues/638
        for node, data in g_out.nodes(data=True):
            if 'osmid' in data:
                data['osmid_original'] = data.pop('osmid')
        # shapefile format is proprietary and deprecated, docs suggests using GeoPackage
        # ox.save_graph_shapefile(g_out, filepath, encoding='utf-8')
        ox.save_graph_geopackage(g_out, filepath, encoding='utf-8')
    return g_


def infer_travel_time_single_threaded(g_, mask, conv_dict, min_z=128, dx=4, dy=4, percentile=90, max_speed_band=-2,
                                      use_weighted_mean=True, variable_edge_speed=False, verbose=False):
    """
    Get an estimate of the average speed and travel time of each edge
    in the graph from the mask and conversion dictionary
    For each edge, get the geometry in pixel coordinates
      For each point, get the nearest neighbors in the maks and infer
      the local speed
    """
    mph_to_mps = 0.44704  # miles per hour to meters per second
    for i, (u, v, edge_data) in enumerate(g_.edges(data=True)):
        if verbose:
            logger.info("\n" + str(i) + " " + str(u) + " " + str(v) + " " + str(edge_data))
        if (i % 1000) == 0:
            logger.info(str(i) + " / " + str(len(g_.edges())) + " edges")
        tot_hours, mean_speed_mph, length_miles = get_edge_time_properties(mask, edge_data, conv_dict, min_z=min_z,
                                                                           dx=dx, dy=dy, percentile=percentile,
                                                                           max_speed_band=max_speed_band,
                                                                           use_weighted_mean=use_weighted_mean,
                                                                           variable_edge_speed=variable_edge_speed,
                                                                           verbose=verbose)
        # Update edges
        edge_data['Travel Time (h)'] = tot_hours
        edge_data['inferred_speed_mph'] = np.round(mean_speed_mph, 2)
        edge_data['length_miles'] = length_miles
        edge_data['inferred_speed_mps'] = np.round(mean_speed_mph * mph_to_mps, 2)
        edge_data['travel_time_s'] = np.round(3600. * tot_hours, 3)
    return g_


def add_travel_time_dir(graph_dir, mask_dir, conv_dict, graph_dir_out, min_z=128, dx=4, dy=4, percentile=90,
                        max_speed_band=-2, use_weighted_mean=True, variable_edge_speed=False, mask_prefix='',
                        save_geopackages=True, n_threads=12, verbose=False):
    """Update graph properties to include travel time for entire directory."""
    t0 = time.time()
    pickle_protocol = 4  # 4 is most recent, python 2.7 can't read 4
    logger.info("Updating graph properties to include travel time")
    logger.info("  Writing to: " + str(graph_dir_out))
    os.makedirs(graph_dir_out, exist_ok=True)
    image_names = sorted([z for z in os.listdir(mask_dir) if z.endswith('.tif')])
    n_files = len(image_names)
    n_threads = min(n_threads, n_files)
    params = []
    for i, image_name in enumerate(image_names):
        im_root = image_name.split('.')[0]
        if len(mask_prefix) > 0:
            im_root = im_root.split(mask_prefix)[-1]
        out_file = os.path.join(graph_dir_out, im_root + '.gpickle')
        if (i % 1) == 0:
            logger.info("\n" + str(i + 1) + " / " + str(len(image_names)) + " " + image_name + " " + im_root)
        mask_path = os.path.join(mask_dir, image_name)
        graph_path = os.path.join(graph_dir, im_root + '.gpickle')
        if not os.path.exists(graph_path):
            logger.info("  " + str(i) + "DNE, skipping: " + str(graph_path))
            continue
        if verbose:
            logger.info("mask_path: " + mask_path)
            logger.info("graph_path: " + graph_path)
        mask = skimage.io.imread(mask_path)
        g_raw = nx.read_gpickle(graph_path)
        # See if it's empty
        if len(g_raw.nodes()) == 0:
            nx.write_gpickle(g_raw, out_file, protocol=pickle_protocol)
            continue
        params.append((g_raw, mask, conv_dict, min_z, dx, dy, percentile, max_speed_band, use_weighted_mean,
                       variable_edge_speed, verbose, out_file, save_geopackages, im_root, graph_dir_out))
    # Execute
    if n_threads > 1:
        pool = Pool(n_threads)
        pool.map(infer_travel_time, params)
    else:
        infer_travel_time(params[0])
    tf = time.time()
    print("Time to infer speed:", tf - t0, "seconds")
    return


class SpaceNet5InferSpeedTask(BaseTask):
    """
    Implements the functionality of step 06 in the CRESI framework.
    """
    schema = SpaceNet5InferSpeedTaskSchema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
            config : dict
        """
        super().__init__(model, config)

    def run(self):
        """
        Implements the main logic behind the task.
        """
        t0 = time.time()
        percentile = 85  # percentile filter (default = 85)
        dx, dy = 6, 6  # nearest neighbors patch size  (default = (4, 4))
        min_z = 128  # min z value to consider a hit (default = 128)
        n_plots = 0
        # Set speed bands, assume a total channel is appended to the speed channels
        if self.config.skeleton_band > 0:
            max_speed_band = self.config.skeleton_band - 1
        else:
            max_speed_band = self.config.num_channels - 1
        save_geopackages = True
        use_weighted_mean = True
        variable_edge_speed = False
        verbose = False
        n_threads = 12
        # Input directories
        res_root_dir = os.path.join(self.config.path_results_root, self.config.test_results_dir)
        graph_dir = os.path.join(res_root_dir, self.config.graph_dir)
        # Output dirs
        graph_speed_dir = os.path.join(res_root_dir, self.config.graph_dir + '_speed')
        os.makedirs(graph_speed_dir, exist_ok=True)
        logger.info("graph_speed_dir: " + graph_speed_dir)
        # Speed conversion dataframes (see _speed_data_prep.ipynb)
        speed_conversion_file = self.config.speed_conversion_file
        # Get the conversion diction between pixel mask values and road speed (mph)
        if self.config.num_classes > 1:
            conv_df, conv_dict = load_speed_conversion_dict_binned(speed_conversion_file)
        else:
            conv_df, conv_dict = load_speed_conversion_dict_continuous(speed_conversion_file)
        logger.info("speed conv_dict: " + str(conv_dict))
        # Add travel time to entire dir
        add_travel_time_dir(graph_dir, self.config.masks_dir, conv_dict, graph_speed_dir, min_z=min_z, dx=dx, dy=dy,
                            percentile=percentile, max_speed_band=max_speed_band, use_weighted_mean=use_weighted_mean,
                            variable_edge_speed=variable_edge_speed, save_geopackages=save_geopackages,
                            n_threads=n_threads, verbose=verbose)
        t1 = time.time()
        logger.info("Time to execute add_travel_time_dir(): {x} seconds".format(x=t1 - t0))
        # Plot a few
        if n_plots > 0:
            logger.info("\nPlot a few...")
            # Plotting
            fig_size = (12, 12)
            # Best colors
            node_color, edge_color = '#cc9900', '#ffbf00'  # gold
            default_node_size = 2
            plot_width_key, plot_width_multiplier = 'inferred_speed_mph', 0.085
            # Define output dir
            graph_speed_plots_dir = os.path.join(res_root_dir, self.config.graph_dir + '_speed_plots')
            os.makedirs(graph_speed_plots_dir, exist_ok=True)
            # Plot graph on image (with width proportional to speed)
            path_images = self.config.test_data_refined_dir
            image_list = [z for z in os.listdir(path_images) if z.endswith('tif')]
            if len(image_list) > n_plots:
                image_names = np.random.choice(image_list, n_plots)
            else:
                image_names = sorted(image_list)
            for i, image_name in enumerate(image_names):
                if i > 10:
                    break
                image_path = os.path.join(path_images, image_name)
                logger.info("\n\nPlotting: " + image_name + "  " + image_path)
                pkl_path = os.path.join(graph_speed_dir, image_name.split('.')[0] + '.gpickle')
                logger.info("   pkl_path: " + pkl_path)
                if not os.path.exists(pkl_path):
                    logger.info("    missing pkl: " + pkl_path)
                    continue
                g = nx.read_gpickle(pkl_path)
                fig_name = os.path.join(graph_speed_plots_dir, image_name)
                fig_name = fig_name.replace('.tif', '.png')
                _ = plot_graph_on_im_yuge(g, image_path, fig_size=fig_size, show_end_nodes=True,
                                          default_node_size=default_node_size,
                                          width_key=plot_width_key, width_multiplier=plot_width_multiplier,
                                          node_color=node_color, edge_color=edge_color, title=image_name,
                                          fig_name=fig_name, verbose=True, super_verbose=verbose)
        t2 = time.time()
        logger.info("Time to execute add_travel_time_dir(): {x} seconds".format(x=t1 - t0))
        logger.info("Time to make plots: {x} seconds".format(x=t2 - t1))
        logger.info("Total time: {x} seconds".format(x=t2 - t0))
        print("Total time: {x} seconds".format(x=t2 - t0))
