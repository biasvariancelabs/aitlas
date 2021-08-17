"""
Notes
-----
    Based on the implementation at:
        https://github.com/CosmiQ/cresi/blob/master/cresi/08_plot_graph_plus_im.py
"""
import logging
import os
import random
import time

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx.settings as ox_settings

# cv2 can't load large files, so need to import skimage too
import skimage.io
from aitlas.base import BaseTask
from aitlas.tasks.schemas import SpaceNet5PlotGraphOverImageTaskSchema
from matplotlib.collections import LineCollection
from osmnx.utils import log
from shapely import wkt
from shapely.geometry import LineString, Point


# Create or get the logger
logger = logging.getLogger(__name__)
# Set log level
logger.setLevel(logging.INFO)


def graph_to_geo_dfs_pix(
    g, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True
):
    """
    Convert a graph into node and/or edge GeoDataFrames

    Parameters
    ----------
    g : networkx MultiDiGraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes

    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """
    if not (nodes or edges):
        raise ValueError("You must request nodes or edges, or both.")
    result = list()
    if nodes:
        start_time = time.time()
        nodes = {node: data for node, data in g.nodes(data=True)}
        gdf_nodes = gpd.GeoDataFrame(nodes).T
        if node_geometry:
            gdf_nodes["geometry_pix"] = gdf_nodes.apply(
                lambda row: Point(row["x_pix"], row["y_pix"]), axis=1
            )
        gdf_nodes.crs = g.graph["crs"]
        gdf_nodes.gdf_name = "{}_nodes".format(g.graph["name"])
        gdf_nodes["osmid"] = gdf_nodes["osmid"].astype(np.int64).map(str)
        result.append(gdf_nodes)
        log(
            'Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(
                gdf_nodes.gdf_name, time.time() - start_time
            )
        )
    if edges:
        start_time = time.time()
        # Create a list to hold our edges, then loop through each edge in the graph
        edges = []
        for u, v, key, data in g.edges(keys=True, data=True):
            # For each edge, add key and all attributes in data dict to the edge_details
            edge_details = {"u": u, "v": v, "key": key}
            for attr_key in data:
                edge_details[attr_key] = data[attr_key]
            # If edge doesn't already have a geometry attribute, create one now
            if "geometry_pix" not in data:
                if fill_edge_geometry:
                    point_u = Point((g.nodes[u]["x_pix"], g.nodes[u]["y_pix"]))
                    point_v = Point((g.nodes[v]["x_pix"], g.nodes[v]["y_pix"]))
                    edge_details["geometry_pix"] = LineString([point_u, point_v])
                else:
                    edge_details["geometry_pix"] = np.nan
            edges.append(edge_details)
        # Create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        gdf_edges.crs = g.graph["crs"]
        gdf_edges.gdf_name = "{}_edges".format(g.graph["name"])
        result.append(gdf_edges)
        log(
            'Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(
                gdf_edges.gdf_name, time.time() - start_time
            )
        )
    if len(result) > 1:
        return tuple(result)
    else:
        return result[0]


def plot_graph_pix(
    g,
    image=None,
    bbox=None,
    fig_height=6,
    fig_width=None,
    margin=0.02,
    axis_off=True,
    equal_aspect=False,
    bg_color="w",
    show=True,
    save=False,
    close=True,
    file_format="png",
    filename="temp",
    default_dpi=300,
    annotate=False,
    node_color="#66ccff",
    node_size=15,
    node_alpha=1,
    node_edge_color="none",
    node_z_order=1,
    edge_color="#999999",
    edge_line_width=1,
    edge_alpha=1,
    edge_color_key="speed_mph",
    color_dict=None,
    edge_width_key="speed_mph",
    edge_width_multiplier=1.0 / 25,
    use_geom=True,
    invert_x_axis=False,
    invert_y_axis=False,
    fig=None,
    ax=None,
):
    """
    Plot a networkx spatial graph.

    Parameters
    ----------
    g : networkx MultiDiGraph
    image : image
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from spatial extents of data.
        If passing a bbox, you probably also want to pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    equal_aspect : bool
        if True set the axis aspect ratio equal
    bg_color : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    default_dpi : int
        the resolution of the image file if saving (may get altered for
        large images)
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edge_color : string
        the color of the node's marker's border
    node_z_order : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_line_width : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    edge_color_key : str
    color_dict : dict
    edge_width_key : str
        optional: key in edge properties to determine edge width, supersedes edge_linewidth, default to "speed_mph"
    edge_width_multiplier : float
        factor to rescale width for plotting, default to 1./25, which gives a line width of 1 for 25 mph speed limit.
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw geographically accurate edges,
        rather than just lines straight from node to node
    invert_x_axis : bool
    invert_y_axis : bool
    fig
    ax

    Returns
    -------
    fig, ax : tuple
    """
    if color_dict is None:
        color_dict = {}
    log("Begin plotting the graph...")
    node_xs = [float(x) for _, x in g.nodes(data="x_pix")]
    node_ys = [float(y) for _, y in g.nodes(data="y_pix")]
    # Get north, south, east, west values either from bbox parameter or from the spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_geo_dfs_pix(g, nodes=False, fill_edge_geometry=True)
        west, south, east, north = gpd.GeoSeries(edges["geometry_pix"]).total_bounds
    else:
        north, south, east, west = bbox
    # If caller did not pass in a fig_width, calculate it proportionately
    # from the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north - south) / (east - west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio
    # Create the figure and axis
    if image is not None:
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(image)
    else:
        if fig is None and ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bg_color)
        ax.set_facecolor(bg_color)
    # Draw the edges as lines from node to node
    start_time = time.time()
    lines = []
    widths = []
    edge_colors = []
    for u, v, data in g.edges(keys=False, data=True):
        if "geometry_pix" in data and use_geom:
            # If it has a geometry attribute (a list of line segments), add them to the list of lines to plot
            xs, ys = data["geometry_pix"].xy
            lines.append(list(zip(xs, ys)))
        else:
            # If it doesn't have a geometry attribute, the edge is a straight line from node to node
            x1 = g.nodes[u]["x_pix"]
            y1 = g.nodes[u]["y_pix"]
            x2 = g.nodes[v]["x_pix"]
            y2 = g.nodes[v]["y_pix"]
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
        # Get widths
        if edge_width_key in data.keys():
            width = int(np.rint(data[edge_width_key] * edge_width_multiplier))
        else:
            width = edge_line_width
        widths.append(width)
        if edge_color_key and color_dict:
            color_key_val = int(data[edge_color_key])
            edge_colors.append(color_dict[color_key_val])
        else:
            edge_colors.append(edge_color)
    # Add the lines to the axis as a LineCollection
    lc = LineCollection(
        lines, colors=edge_colors, linewidths=widths, alpha=edge_alpha, zorder=2
    )
    ax.add_collection(lc)
    log("Drew the graph edges in {:,.2f} seconds".format(time.time() - start_time))
    # Scatter plot the nodes
    ax.scatter(
        node_xs,
        node_ys,
        s=node_size,
        c=node_color,
        alpha=node_alpha,
        edgecolor=node_edge_color,
        zorder=node_z_order,
    )
    # Set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))
    # Configure axis appearance
    x_axis = ax.get_xaxis()
    y_axis = ax.get_yaxis()
    x_axis.get_major_formatter().set_useOffset(False)
    y_axis.get_major_formatter().set_useOffset(False)
    # If axis_off,
    # turn off the axis display set the margins to zero and point the ticks in so there's no space around the plot.
    if axis_off:
        ax.axis("off")
        ax.margins(0)
        ax.tick_params(which="both", direction="in")
        x_axis.set_visible(False)
        y_axis.set_visible(False)
        fig.canvas.draw()
    if equal_aspect:
        # Make everything square
        ax.set_aspect("equal")
        fig.canvas.draw()
    else:
        # If the graph is not projected, conform the aspect ratio to not stretch the plot
        if g.graph["crs"] == ox_settings.default_crs:
            cos_lat = np.cos((min(node_ys) + max(node_ys)) / 2.0 / 180.0 * np.pi)
            ax.set_aspect(1.0 / cos_lat)
            fig.canvas.draw()
    # Annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in g.nodes(data=True):
            ax.annotate(node, xy=(data["x_pix"], data["y_pix"]))
    # Update dpi, if image
    if image is not None:
        # mpl can handle a max of 2^29 pixels, or 23170 on a side
        # Recompute max_dpi
        max_dpi = int(23000 / max(fig_height, fig_width))
        h, w = image.shape[:2]
        # Try to set dpi to native resolution of imagery
        desired_dpi = max(default_dpi, 1.0 * h / fig_height)
        dpi = int(np.min([max_dpi, desired_dpi]))
    else:
        dpi = default_dpi
    # Save and show the figure as specified
    fig, ax = save_and_show(
        fig,
        ax,
        save,
        show,
        close,
        filename,
        file_format,
        dpi,
        axis_off,
        invert_x_axis=invert_x_axis,
        invert_y_axis=invert_y_axis,
    )
    return fig, ax


def plot_graph_route_pix(
    g,
    route,
    image=None,
    bbox=None,
    fig_height=6,
    fig_width=None,
    margin=0.02,
    bg_color="w",
    axis_off=True,
    show=True,
    save=False,
    close=True,
    file_format="png",
    filename="temp",
    default_dpi=300,
    annotate=False,
    node_color="#999999",
    node_size=15,
    node_alpha=1,
    node_edge_color="none",
    node_z_order=1,
    edge_color="#999999",
    edge_line_width=1,
    edge_alpha=1,
    edge_color_key="speed_mph",
    color_dict=None,
    edge_width_key="speed_mph",
    edge_width_multiplier=1.0 / 25,
    use_geom=True,
    origin_point=None,
    destination_point=None,
    route_color="r",
    route_line_width=4,
    route_alpha=0.5,
    orig_destination_node_alpha=0.5,
    orig_destination_node_size=100,
    orig_destination_node_color="r",
    invert_x_axis=False,
    invert_y_axis=True,
    fig=None,
    ax=None,
):
    """
    Plot a route along a networkx spatial graph.

    Parameters
    ----------
    g : networkx MultiDiGraph
    route : list
        the route as a list of nodes
    image : image
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from spatial extents of data.
        If passing a bbox, you probably also want to pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    bg_color : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    default_dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edge_color : string
        the color of the node's marker's border
    node_z_order : int
        z_order to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_line_width : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    edge_color_key : str
    color_dict : dict
    edge_width_key : str
    edge_width_multiplier : float
        factor to rescale width for plotting, default to 1./25, which gives a line width of 1 for 25 mph speed limit.
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    origin_point : tuple
        optional, an origin (lat, lon) point to plot instead of the origin node
    destination_point : tuple
        optional, a destination (lat, lon) point to plot instead of the
        destination node
    route_color : string
        the color of the route
    route_line_width : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    orig_destination_node_alpha : float
        the opacity of the origin and destination nodes
    orig_destination_node_size : int
        the size of the origin and destination nodes
    orig_destination_node_color : string
        the color of the origin and destination nodes
        (can be a string or list with (origin_color, dest_color))
        of nodes
    invert_x_axis : bool
    invert_y_axis : bool
    fig
    ax

    Returns
    -------
    fig, ax : tuple
    """
    if color_dict is None:
        color_dict = {}
    # Plot the graph but not the route
    fig, ax = plot_graph_pix(
        g,
        image=image,
        bbox=bbox,
        fig_height=fig_height,
        fig_width=fig_width,
        margin=margin,
        axis_off=axis_off,
        bg_color=bg_color,
        show=False,
        save=False,
        close=False,
        filename=filename,
        default_dpi=default_dpi,
        annotate=annotate,
        node_color=node_color,
        node_size=node_size,
        node_alpha=node_alpha,
        node_edge_color=node_edge_color,
        node_z_order=node_z_order,
        edge_color_key=edge_color_key,
        color_dict=color_dict,
        edge_color=edge_color,
        edge_line_width=edge_line_width,
        edge_alpha=edge_alpha,
        edge_width_key=edge_width_key,
        edge_width_multiplier=edge_width_multiplier,
        use_geom=use_geom,
        fig=fig,
        ax=ax,
    )
    # The origin and destination nodes are the first and last nodes in the route
    origin_node = route[0]
    destination_node = route[-1]
    if origin_point is None or destination_point is None:
        # If caller didn't pass points, use the first and last node in route as origin/destination
        origin_destination_ys = (
            g.nodes[origin_node]["y_pix"],
            g.nodes[destination_node]["y_pix"],
        )
        origin_destination_xs = (
            g.nodes[origin_node]["x_pix"],
            g.nodes[destination_node]["x_pix"],
        )
    else:
        # Otherwise, use the passed points as origin/destination
        origin_destination_xs = (origin_point[0], destination_point[0])
        origin_destination_ys = (origin_point[1], destination_point[1])
    # Scatter the origin and destination points
    ax.scatter(
        origin_destination_xs,
        origin_destination_ys,
        s=orig_destination_node_size,
        c=orig_destination_node_color,
        alpha=orig_destination_node_alpha,
        edgecolor=node_edge_color,
        zorder=4,
    )
    # Plot the route lines
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # If there are parallel edges, select the shortest in length
        data = min(g.get_edge_data(u, v).values(), key=lambda x: x["length"])
        # If it has a geometry attribute (i.e., a list of line segments)
        if "geometry_pix" in data and use_geom:
            # Add them to the list of lines to plot
            xs, ys = data["geometry_pix"].xy
            lines.append(list(zip(xs, ys)))
        else:
            # If it doesn't have a geometry attribute, the edge is a straight line from node to node
            x1 = g.nodes[u]["x_pix"]
            y1 = g.nodes[u]["y_pix"]
            x2 = g.nodes[v]["x_pix"]
            y2 = g.nodes[v]["y_pix"]
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    # Add the lines to the axis as a LineCollection
    lc = LineCollection(
        lines,
        colors=route_color,
        linewidths=route_line_width,
        alpha=route_alpha,
        zorder=3,
    )
    ax.add_collection(lc)
    # Update dpi, if image
    if image is not None:
        # mpl can handle a max of 2^29 pixels, or 23170 on a side
        # Recompute max_dpi
        max_dpi = int(23000 / max(fig_height, fig_width))
        h, w = image.shape[:2]
        # Try to set dpi to native resolution of imagery
        desired_dpi = max(default_dpi, 1.0 * h / fig_height)
        dpi = int(np.min([max_dpi, desired_dpi]))
    # Save and show the figure as specified
    fig, ax = save_and_show(
        fig,
        ax,
        save,
        show,
        close,
        filename,
        file_format,
        dpi,
        axis_off,
        invert_y_axis=invert_y_axis,
        invert_x_axis=invert_x_axis,
    )
    return fig, ax


def save_and_show(
    fig,
    ax,
    save,
    show,
    close,
    filename,
    file_format,
    dpi,
    axis_off,
    tight_layout=False,
    invert_x_axis=False,
    invert_y_axis=True,
    verbose=False,
):
    """
    Save a figure to disk and show it, as specified.
    Assume filename holds entire path to file.

    Parameters
    ----------
    fig : figure
    ax : axis
    save : bool
        whether to save the figure to disk or not
    show : bool
        whether to display the figure or not
    close : bool
        close the figure (only if show equals False) to prevent display
    filename : string
        the name of the file to save
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    dpi : int
        the resolution of the image file if saving
    axis_off : bool
        if True matplotlib axis was turned off by plot_graph so constrain the
        saved figure's extent to the interior of the axis
    tight_layout : bool
    invert_x_axis : bool
    invert_y_axis : bool
    verbose : bool

    Returns
    -------
    fig, ax : tuple
    """
    if invert_y_axis:
        ax.invert_yaxis()
    if invert_x_axis:
        ax.invert_xaxis()
    # Save the figure if specified
    if save:
        start_time = time.time()
        # Create the save folder if it doesn't already exist
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        path_filename = filename
        if file_format == "svg":
            # If the file_format is svg, prep the fig / ax a bit for saving
            ax.axis("off")
            ax.set_position([0, 0, 1, 1])
            ax.patch.set_alpha(0.0)
            fig.patch.set_alpha(0.0)
            fig.savefig(
                path_filename,
                bbox_inches=0,
                format=file_format,
                facecolor=fig.get_facecolor(),
                transparent=True,
            )
        else:
            if axis_off:
                # If axis is turned off, constrain the saved figure's extent to the interior of the axis
                extent = ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()
                )
            else:
                extent = "tight"
            if tight_layout:
                # extent = 'tight'
                fig.gca().set_axis_off()
                fig.subplots_adjust(
                    top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
                )
                plt.margins(0, 0)
                fig.savefig(
                    path_filename,
                    dpi=dpi,
                    bbox_inches=extent,
                    format=file_format,
                    facecolor=fig.get_facecolor(),
                    transparent=True,
                    pad_inches=0,
                )
            else:
                fig.savefig(
                    path_filename,
                    dpi=dpi,
                    bbox_inches=extent,
                    format=file_format,
                    facecolor=fig.get_facecolor(),
                    transparent=True,
                )
        if verbose:
            print(
                "Saved the figure to disk in {:,.2f} seconds".format(
                    time.time() - start_time
                )
            )
    # Show the figure if specified
    if show:
        start_time = time.time()
        plt.show()
        if verbose:
            print("Showed the plot in {:,.2f} seconds".format(time.time() - start_time))
    # If show=False, close the figure if close=True to prevent display
    elif close:
        plt.close()
    return fig, ax


def color_func(speed):
    """Define colors (yellow to red color ramp)."""
    if speed < 15:
        color = "#ffffb2"
    elif 15 <= speed < 25:
        color = "#ffe281"
    elif 25 <= speed < 35:
        color = "#fec357"
    elif 35 <= speed < 45:
        color = "#fe9f45"
    elif 45 <= speed < 55:
        color = "#fa7634"
    elif 55 <= speed < 65:
        color = "#f24624"
    elif 65 <= speed < 75:
        color = "#da2122"
    else:  # speed >= 75
        color = "#bd0026"
    return color


def make_color_dict_list(max_speed=80, verbose=False):
    color_dict = {}
    color_list = []
    for speed in range(max_speed):
        c = color_func(speed)
        color_dict[speed] = c
        color_list.append(c)
    if verbose:
        print("color_dict:", color_dict)
        print("color_list:", color_list)
    return color_dict, color_list


class SpaceNet5PlotGraphOverImageTask(BaseTask):
    """
    Implements the functionality of step 08 in the CRESI framework.
    """

    schema = SpaceNet5PlotGraphOverImageTaskSchema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
            config : Config
        """
        super().__init__(model, config)

    def run(self):
        """
        Implements the main logic of the task.

        Plotting adapted from:
            https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py"""
        # Output files
        res_root_dir = os.path.join(
            self.config.path_results_root, self.config.test_results_dir
        )
        path_images_8bit = os.path.join(self.config.test_data_refined_dir)
        graph_dir = os.path.join(res_root_dir, self.config.graph_dir + "_speed")
        out_dir = graph_dir.strip() + "_plots"
        # Initialize variables
        save_only_route_png = False
        fig_height = 12
        fig_width = 12
        node_color = "#66ccff"
        node_size = 0.4
        node_alpha = 0.6
        edge_color = "#bfefff"
        edge_line_width = 0.5
        edge_alpha = 0.6
        edge_color_key = "inferred_speed_mph"
        shuffle = True
        invert_x_axis = False
        invert_y_axis = False
        # Iterate through images and graphs, plot routes
        im_list = sorted(
            [z for z in os.listdir(path_images_8bit) if z.endswith(".tif")]
        )
        # Modify this variable to control the number of plots produced
        max_plots = len(im_list)
        if shuffle:
            random.shuffle(im_list)
        for i, im_root in enumerate(im_list):
            if not im_root.endswith(".tif"):
                continue
            if i >= max_plots:
                break
            im_root_no_ext = im_root.split(".tif")[0]
            im_file = os.path.join(path_images_8bit, im_root)
            graph_pkl = os.path.join(graph_dir, im_root_no_ext + ".gpickle")
            print("\n\n", i, "im_root:", im_root)
            print("  im_file:", im_file)
            print("  graph_pkl:", graph_pkl)
            # gpickle?
            print("Reading gpickle...")
            if not os.path.exists(graph_pkl):
                continue
            G = nx.read_gpickle(graph_pkl)
            # Get one node, check longitude
            node = list(G.nodes())[-1]
            print(node, "random node props:", G.nodes[node])
            if G.nodes[node]["lat"] < 0:
                print("Negative latitude, inverting yaxis for plotting")
                invert_y_axis = True
            # Make sure geometries are not just strings
            print("Make sure geometries are not just strings...")
            for u, v, key, data in G.edges(keys=True, data=True):
                for attr_key in data:
                    if (attr_key == "geometry") and (type(data[attr_key]) == str):
                        data[attr_key] = wkt.loads(data[attr_key])
                    elif (attr_key == "geometry_pix") and (type(data[attr_key]) == str):
                        data[attr_key] = wkt.loads(data[attr_key])
                    else:
                        continue
            # Read in image, cv2 fails on large files
            print("Read in image...")
            try:
                # Convert to rgb (cv2 reads in bgr)
                img_cv2 = cv2.imread(im_file, 1)
                print("img_cv2.shape:", img_cv2.shape)
                im = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            except:
                im = skimage.io.imread(im_file).astype(np.uint8)
            # Set dpi to approximate native resolution
            print("im.shape:", im.shape)
            desired_dpi = int(np.max(im.shape) / np.max([fig_height, fig_width]))
            print("desired dpi:", desired_dpi)
            # Max out dpi at 3500
            dpi = int(np.min([3500, desired_dpi]))
            print("plot dpi:", dpi)
            # Plot graph with image background
            if not save_only_route_png:
                out_file_plot = os.path.join(out_dir, im_root_no_ext + "_ox_plot.tif")
                print("outfile_plot:", out_file_plot)
                plot_graph_pix(
                    G,
                    im,
                    fig_height=fig_height,
                    fig_width=fig_width,
                    node_size=int(node_size),
                    node_alpha=node_alpha,
                    node_color=node_color,
                    edge_line_width=edge_line_width,
                    edge_alpha=edge_alpha,
                    edge_color=edge_color,
                    filename=out_file_plot,
                    default_dpi=dpi,
                    edge_color_key="",
                    show=False,
                    save=True,
                    invert_y_axis=invert_y_axis,
                    invert_x_axis=invert_x_axis,
                )
                # Plot with speed
                out_file_plot_speed = os.path.join(
                    out_dir, im_root_no_ext + "_ox_plot_speed.tif"
                )
                print("outfile_plot_speed:", out_file_plot_speed)
                color_dict, color_list = make_color_dict_list()
                plot_graph_pix(
                    G,
                    im,
                    fig_height=fig_height,
                    fig_width=fig_width,
                    node_size=int(node_size),
                    node_alpha=node_alpha,
                    node_color=node_color,
                    edge_line_width=edge_line_width,
                    edge_alpha=edge_alpha,
                    edge_color=edge_color,
                    filename=out_file_plot_speed,
                    default_dpi=dpi,
                    show=False,
                    save=True,
                    invert_y_axis=invert_y_axis,
                    invert_x_axis=invert_x_axis,
                    edge_color_key=edge_color_key,
                    color_dict=color_dict,
                )
