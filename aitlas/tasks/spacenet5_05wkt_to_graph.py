import logging
import os
import time
from multiprocessing.pool import Pool

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import shapely.ops
import shapely.wkt
import utm
from math import sqrt
from osgeo import gdal, ogr, osr
from shapely.geometry import Point, LineString

from aitlas.base import BaseTask
from .schemas import SpaceNet5WktToGraphTaskSchema

# Create or get the logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.INFO)


def clean_sub_graphs(g_, min_length=300, max_nodes_to_skip=20, weight='length_pix', verbose=True, super_verbose=False):
    """
    Remove sub-graphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length (this step great improves processing time).
    """
    if len(g_.nodes()) == 0:
        return g_
    if verbose:
        print("Running clean_sub_graphs...")
    try:
        # https://stackoverflow.com/questions/61154740/attributeerror-module-networkx-has-no-attribute-connected-component-subgraph
        sub_graphs = [g_.subgraph(c) for c in nx.connected_components(g_)]
    except:
        sub_graph_nodes = nx.connected_components(g_)
        sub_graphs = [g_.subgraph(c).copy() for c in sub_graph_nodes]
    if verbose:
        print("  sub_graph node count:", [len(z.nodes) for z in sub_graphs])
    bad_nodes = []
    if verbose:
        print("  len(G_.nodes()):", len(g_.nodes()))
        print("  len(G_.edges()):", len(g_.edges()))
    if super_verbose:
        print("G_.nodes:", g_.nodes())
        edge_tmp = g_.edges()[np.random.randint(len(g_.edges()))]
        print(edge_tmp, "G.edge props:", g_.edge[edge_tmp[0]][edge_tmp[1]])
    for G_sub in sub_graphs:
        # Don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes())
                print("  all_lengths:", all_lengths)
            # Get all lengths
            lens = []
            for u in all_lengths.keys():
                v = all_lengths[u]
                for u_prime in v.keys():
                    v_prime = v[u_prime]
                    lens.append(v_prime)
                    if super_verbose:
                        print("  u, v", u, v)
                        print("    u_prime, v_prime:", u_prime, v_prime)
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())
    # Remove bad_nodes
    g_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        print(" len(G'.nodes()):", len(g_.nodes()))
        print(" len(G'.edges()):", len(g_.edges()))
    if super_verbose:
        print("  G_.nodes:", g_.nodes())
    return g_


def wkt_list_to_nodes_edges(wkt_list, node_iter=10000, edge_iter=10000):
    """
    Convert wkt list to nodes and edges. Make an edge between each node in linestring.
    Since one linestring may contain multiple edges, this is the safest approach.
    """
    node_loc_set = set()  # set of edge locations
    node_loc_dic = {}  # key = node idx, val = location
    node_loc_dic_rev = {}  # key = location, val = node idx
    edge_loc_set = set()  # set of edge locations
    edge_dic = {}  # edge properties
    for i, l_string in enumerate(wkt_list):
        # Get l_string properties
        shape = shapely.wkt.loads(l_string)
        xs, ys = shape.coords.xy
        # Iterate through coordinates in line to create edges between every point
        for j, (x, y) in enumerate(zip(xs, ys)):
            loc = (x, y)
            # For first item just make node, not edge
            if j == 0:
                # If not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node_iter += 1
            # If not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j - 1], ys[j - 1])
                prev_node = node_loc_dic_rev[prev_loc]
                # If new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # If seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]
                # Add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # Shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    print("Oops, edge already seen, returning:", edge_loc)
                    return
                # Get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # Edge length is the difference of the two projected lengths along the linestring
                edge_length = abs(proj - proj_prev)
                # Make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt
                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length_pix': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry_pix': line_out,
                              'osmid': i}
                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1
    return node_loc_dic, edge_dic


def nodes_edges_to_g(node_loc_dic, edge_dic, name='glurp'):
    """Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx graph."""
    g = nx.MultiDiGraph()
    # Set graph crs and name
    g.graph = {'name': name, 'crs': {'init': 'epsg:4326'}}
    # Add nodes
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x_pix': val[0],
                     'y_pix': val[1]}
        g.add_node(key, **attr_dict)
    # Add edges
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']
        if type(attr_dict['start_loc_pix']) == list:
            return
        g.add_edge(u, v, **attr_dict)
    return g.to_undirected()


def pixel_to_geo_coord(params):
    """from spacenet geotools"""
    source_sr = ''
    geom_transform = ''
    target_sr = osr.SpatialReference()
    target_sr.ImportFromEPSG(4326)
    identifier, x_pix, y_pix, input_raster = params
    if target_sr == '':
        perform_reprojection = False
        target_sr = osr.SpatialReference()
        target_sr.ImportFromEPSG(4326)
    else:
        perform_reprojection = True
    if geom_transform == '':
        src_raster = gdal.Open(input_raster)
        geom_transform = src_raster.GetGeoTransform()
        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(src_raster.GetProjectionRef())
    geom = ogr.Geometry(ogr.wkbPoint)
    x_origin = geom_transform[0]
    y_origin = geom_transform[3]
    pixel_width = geom_transform[1]
    pixel_height = geom_transform[5]
    x_coord = (x_pix * pixel_width) + x_origin
    y_coord = (y_pix * pixel_height) + y_origin
    geom.AddPoint(x_coord, y_coord)
    if perform_reprojection:
        if source_sr == '':
            src_raster = gdal.Open(input_raster)
            source_sr = osr.SpatialReference()
            source_sr.ImportFromWkt(src_raster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(source_sr, target_sr)
        geom.Transform(coord_trans)
    return {identifier: (geom.GetX(), geom.GetY())}


def get_node_geo_coordinates(g, im_file, fix_utm_zone=True, n_threads=12, verbose=False):
    # Get pixel params
    params = []
    nn = len(g.nodes())
    for i, (n, attr_dict) in enumerate(g.nodes(data=True)):
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        params.append((n, x_pix, y_pix, im_file))
    if verbose:
        print("node params[:5]:", params[:5])
    n_threads = min(n_threads, nn)
    # Execute
    print("Computing geo coords for nodes (" + str(n_threads) + " threads)...")
    if n_threads > 1:
        pool = Pool(n_threads)
        coordinate_dict_list = pool.map(pixel_to_geo_coord, params)
    else:
        coordinate_dict_list = list()
        for parm in params:
            coordinate_dict_list.append(pixel_to_geo_coord(parm))
    # Combine the disparate dicts
    coordinate_dict = dict()
    for d in coordinate_dict_list:
        for k, v in d.items():
            coordinate_dict.setdefault(k, list()).append(v)
    if verbose:
        print("  nodes: list(coordinate_dict)[:5]:", list(coordinate_dict)[:5])
    # Update data
    print("Updating data properties")
    utm_letter = 'Oooops'
    for i, (n, attr_dict) in enumerate(g.nodes(data=True)):
        if verbose and ((i % 5000) == 0):
            print(i, "/", nn, "node:", n)
        lon, lat = coordinate_dict[n][0]
        # Fix zone
        if i == 0 or fix_utm_zone is False:
            [utm_east, utm_north, utm_zone, utm_letter] = \
                utm.from_latlon(lat, lon)
            if verbose and (i == 0):
                print("utm_letter:", utm_letter)
                print("utm_zone:", utm_zone)
        else:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon, force_zone_number=utm_zone,
                                                          force_zone_letter=utm_letter)
        if lat > 90:
            print("lat > 90, returning:", n, attr_dict)
            return
        attr_dict['lon'] = lon
        attr_dict['lat'] = lat
        attr_dict['utm_east'] = utm_east
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        attr_dict['utm_north'] = utm_north
        attr_dict['x'] = lon
        attr_dict['y'] = lat
        if verbose and ((i % 5000) == 0):
            print("  node, attr_dict:", n, attr_dict)
    return g


def convert_pix_lstring_to_geo(params):
    """
    Convert linestring in pixel coordinates to geo coordinates.
    If zone or letter changes in the middle of line, it's all screwed up, so force zone and letter based on first point.
    (latitude, longitude, force_zone_number=None, force_zone_letter=None)
    Or just force utm zone and letter explicitly.
    """
    identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
    shape = shapely.wkt.loads(geom_pix_wkt)
    x_pixels, y_pixels = shape.coords.xy
    lon_lat_coordinates = []
    utm_coordinates = []
    for i, (x, y) in enumerate(zip(x_pixels, y_pixels)):
        params_tmp = ('tmp', x, y, im_file)
        tmp_dict = pixel_to_geo_coord(params_tmp)
        (lon, lat) = list(tmp_dict.values())[0]
        if utm_zone and utm_letter:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon, force_zone_number=utm_zone,
                                                          force_zone_letter=utm_letter)
        else:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
        if verbose:
            print("lat lon, utm_east, utm_north, utm_zone, utm_letter]",
                  [lat, lon, utm_east, utm_north, utm_zone, utm_letter])
        utm_coordinates.append([utm_east, utm_north])
        lon_lat_coordinates.append([lon, lat])
    l_string_lat_lon = LineString([Point(z) for z in lon_lat_coordinates])
    l_string_utm = LineString([Point(z) for z in utm_coordinates])
    return {identifier: (l_string_lat_lon, l_string_utm, utm_zone, utm_letter)}


def get_edge_geo_coordinates(g, im_file, remove_pix_geom=True, fix_utm_zone=True, n_threads=12, verbose=False,
                             super_verbose=False):
    """Get geo coordinates of all edges."""
    # First, get utm letter and zone of first node in graph
    for i, (n, attr_dict) in enumerate(g.nodes(data=True)):
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        if i > 0:
            break
    params_tmp = ('tmp', x_pix, y_pix, im_file)
    print("params_tmp", params_tmp)
    tmp_dict = pixel_to_geo_coord(params_tmp)
    print("tmp_dict:", tmp_dict)
    (lon, lat) = list(tmp_dict.values())[0]
    [_, _, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
    # Now get edge params
    params = []
    ne = len(list(g.edges()))
    for i, (u, v, attr_dict) in enumerate(g.edges(data=True)):
        geom_pix = attr_dict['geometry_pix']
        # identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
        if not fix_utm_zone:
            params.append(((u, v), geom_pix.wkt, im_file, None, None, super_verbose))
        else:
            params.append(((u, v), geom_pix.wkt, im_file, utm_zone, utm_letter, super_verbose))
    if verbose:
        print("edge params[:5]:", params[:5])
    n_threads = min(n_threads, ne)
    # Execute
    print("Computing geo coords for edges (" + str(n_threads) + " threads)...")
    if n_threads > 1:
        pool = Pool(n_threads)
        coordinate_dict_list = pool.map(convert_pix_lstring_to_geo, params)
    else:
        coordinate_dict_list = list()
        for parm in params:
            coordinate_dict_list.append(convert_pix_lstring_to_geo(parm))
    # Combine the disparate dicts
    coordinate_dict = dict()
    for d in coordinate_dict_list:
        for k, v in d.items():
            coordinate_dict.setdefault(k, list()).append(v)
    if verbose:
        print("  edges: list(coordinate_dict)[:5]:", list(coordinate_dict)[:5])
    print("Updating edge data properties")
    for i, (u, v, attr_dict) in enumerate(g.edges(data=True)):
        geom_pix = attr_dict['geometry_pix']
        l_string_lat_lon, l_string_utm, utm_zone, utm_letter = coordinate_dict[(u, v)][0]
        attr_dict['geometry_latlon_wkt'] = l_string_lat_lon.wkt
        attr_dict['geometry_utm_wkt'] = l_string_utm.wkt
        attr_dict['length_latlon'] = l_string_lat_lon.length
        attr_dict['length_utm'] = l_string_utm.length
        attr_dict['length'] = l_string_utm.length
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        if verbose and ((i % 1000) == 0):
            print("   attr_dict_final:", attr_dict)
        # Geometry screws up osmnx.simplify function
        if remove_pix_geom:
            attr_dict['geometry_pix'] = geom_pix.wkt
        # Try actual geometry, not just linestring, this seems necessary for projections
        attr_dict['geometry'] = l_string_lat_lon
        # Ensure utm length isn't excessive
        if l_string_utm.length > 5000:
            print(u, v, "edge length too long:", attr_dict, "returning!")
            return

    return g


def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs((end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1]))
        d = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        return n / d


def rdp(points, epsilon=1):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    d_max = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > d_max:
            index = i
            d_max = d
    if d_max >= epsilon:
        results = rdp(points[:index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results


def wkt_to_g(params):
    """Execute all functions."""
    n_threads_max = 12
    wkt_list, im_file, min_subgraph_length_pix, node_iter, edge_iter, min_spur_length_m, simplify_graph, rdp_epsilon, \
    manually_reproject_nodes, out_file, graph_dir, n_threads, verbose = params
    print("im_file:", im_file)
    pickle_protocol = 4
    t0 = time.time()
    if verbose:
        print("Running wkt_list_to_nodes_edges()...")
    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list, node_iter=node_iter, edge_iter=edge_iter)
    t1 = time.time()
    if verbose:
        print("Time to run wkt_list_to_nodes_egdes():", t1 - t0, "seconds")
    if verbose:
        print("Creating G...")
    g0 = nodes_edges_to_g(node_loc_dic, edge_dic)
    if verbose:
        print("  len(G.nodes():", len(g0.nodes()))
        print("  len(G.edges():", len(g0.edges()))
    t2 = time.time()
    if verbose:
        print("Time to run nodes_edges_to_G():", t2 - t1, "seconds")
    # This graph will have a unique edge for each line segment,
    # meaning that many nodes will have degree 2 and be in the middle of a long edge.
    # So that adding small terminals works better.
    if verbose:
        print("Clean out short subgraphs")
    g1 = clean_sub_graphs(g0, min_length=min_subgraph_length_pix, weight='length_pix', verbose=verbose,
                          super_verbose=False)
    t3 = time.time()
    if verbose:
        print("Time to run clean_sub_graphs():", t3 - t2, "seconds")
    t3 = time.time()
    if len(g1) == 0:
        return g1
    # Geo coordinates
    if im_file:
        if verbose:
            print("Running get_node_geo_coords()...")
        # Let's not over multi-thread a multi-thread
        if n_threads > 1:
            n_threads_tmp = 1
        else:
            n_threads_tmp = n_threads_max
        g1 = get_node_geo_coordinates(g1, im_file, n_threads=n_threads_tmp, verbose=verbose)
        t4 = time.time()
        if verbose:
            print("Time to run get_node_geo_coords():", t4 - t3, "seconds")
        if verbose:
            print("Running get_edge_geo_coords()...")
        # Let's not over multi-thread a multi-thread
        if n_threads > 1:
            n_threads_tmp = 1
        else:
            n_threads_tmp = n_threads_max
        g1 = get_edge_geo_coordinates(g1, im_file, n_threads=n_threads_tmp, verbose=verbose)
        t5 = time.time()
        if verbose:
            print("Time to run get_edge_geo_coords():", t5 - t4, "seconds")
        if verbose:
            print("pre projection...")
        node = list(g1.nodes())[-1]
        if verbose:
            print(node, "random node props:", g1.nodes[node])
            # Print an edge
            edge_tmp = list(g1.edges())[-1]
            print(edge_tmp, "random edge props:", g1.get_edge_data(edge_tmp[0], edge_tmp[1]))
        if verbose:
            print("projecting graph...")
        g_projected = ox.project_graph(g1)
        # Get geom wkt (for printing/viewing purposes)
        for i, (u, v, attr_dict) in enumerate(g_projected.edges(data=True)):
            if 'geometry' in attr_dict.keys():
                attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt  # attr_dic['geometry'] key error
        if verbose:
            print("post projection...")
            node = list(g_projected.nodes())[-1]
            print(node, "random node props:", g_projected.nodes[node])
            # Print an edge
            edge_tmp = list(g_projected.edges())[-1]
            print(edge_tmp, "random edge props:", g_projected.get_edge_data(edge_tmp[0], edge_tmp[1]))
        t6 = time.time()
        if verbose:
            print("Time to project graph:", t6 - t5, "seconds")
        g_out = g_projected
    else:
        g_out = g0
    if simplify_graph:
        if verbose:
            print("Simplifying graph")
        t7 = time.time()
        # 'geometry' tag breaks simplify, so make it a wkt
        for i, (u, v, attr_dict) in enumerate(g_projected.edges(data=True)):
            if 'geometry' in attr_dict.keys():
                attr_dict['geometry'] = attr_dict['geometry'].wkt
        g0 = ox.simplify_graph(g_out.to_directed())
        g0 = g0.to_undirected()
        # Re-projecting graph screws up lat lon, so convert to string?
        # TODO: Line below throws: ValueError: Geometry must be unprojected to calculate UTM zone
        # g_out = ox.project_graph(g0)
        g_out = g0
        if verbose:
            print("post simplify...")
            node = list(g_out.nodes())[-1]
            print(node, "random node props:", g_out.nodes[node])
            # Print an edge
            edge_tmp = list(g_out.edges())[-1]
            print(edge_tmp, "random edge props:", g_out.get_edge_data(edge_tmp[0], edge_tmp[1]))
        t8 = time.time()
        if verbose:
            print("Time to run simplify graph:", t8 - t7, "seconds")
        # When the simplify function combines edges, it concatenates multiple edge properties into a list.
        # This means that 'geometry_pix' is now a list of geoms.
        # Convert this to a linestring with shaply.ops.linemergeconcats
        # BUG, GOOF, ERROR IN OSMNX PROJECT, SO NEED TO MANUALLY SET X, Y FOR NODES!!??
        if manually_reproject_nodes:
            # Make sure geometry is utm for nodes?
            for i, (n, attr_dict) in enumerate(g_out.nodes(data=True)):
                attr_dict['x'] = attr_dict['utm_east']
                attr_dict['y'] = attr_dict['utm_north']
        if verbose:
            print("Merge 'geometry' linestrings...")
        keys_tmp = ['geometry_wkt', 'geometry_pix', 'geometry_latlon_wkt',
                    'geometry_utm_wkt']
        for key_tmp in keys_tmp:
            if verbose:
                print("Merge", key_tmp, "...")
            for i, (u, v, attr_dict) in enumerate(g_out.edges(data=True)):
                if key_tmp not in attr_dict.keys():
                    continue
                if (i % 10000) == 0:
                    print(i, u, v)
                geom = attr_dict[key_tmp]
                if type(geom) == list:
                    # Check if the list items are wkt strings, if so, create linestrigs
                    if type(geom[0]) == str:  # or (type(geom_pix[0]) == unicode):
                        geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                    # Merge geoms
                    geom_out = shapely.ops.linemerge(geom)
                elif type(geom) == str:
                    geom_out = shapely.wkt.loads(geom)
                else:
                    geom_out = geom
                # Now straighten edge with rdp
                if rdp_epsilon > 0:
                    if verbose and ((i % 10000) == 0):
                        print("  Applying rdp...")
                    coordinates = list(geom_out.coords)
                    new_coordinates = rdp(coordinates, epsilon=rdp_epsilon)
                    geom_out_rdp = LineString(new_coordinates)
                    geom_out_final = geom_out_rdp
                else:
                    geom_out_final = geom_out
                len_out = geom_out_final.length
                # Update edge properties
                attr_dict[key_tmp] = geom_out_final
                # Update length
                if key_tmp == 'geometry_pix':
                    attr_dict['length_pix'] = len_out
                if key_tmp == 'geometry_utm_wkt':
                    attr_dict['length_utm'] = len_out
        # Assign 'geometry' tag to geometry_utm_wkt
        key_tmp = 'geometry_utm_wkt'  # 'geometry_utm_wkt'
        for i, (u, v, attr_dict) in enumerate(g_out.edges(data=True)):
            if verbose and ((i % 10000) == 0):
                print("Create 'geometry' field in edges...")
            line = attr_dict['geometry_utm_wkt']
            if type(line) == str:  # or type(line) == unicode:
                attr_dict['geometry'] = shapely.wkt.loads(line)
            else:
                attr_dict['geometry'] = attr_dict[key_tmp]
            attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt
            # Set length
            attr_dict['length'] = attr_dict['geometry'].length
            # Update wkt_pix?
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt
            # Update 'length_pix'
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])
    # Print a random node and edge
    if verbose:
        node_tmp = list(g_out.nodes())[-1]
        print(node_tmp, "random node props:", g_out.nodes[node_tmp])
        # Print an edge
        edge_tmp = list(g_out.edges())[-1]
        print("random edge props for edge:", edge_tmp, " = ",
              g_out.edges[edge_tmp[0], edge_tmp[1], 0])
    if verbose:
        logger.info("Number of nodes: {}".format(len(g_out.nodes())))
        logger.info("Number of edges: {}".format(len(g_out.edges())))
    g_out.graph['N_nodes'] = len(g_out.nodes())
    g_out.graph['N_edges'] = len(g_out.edges())
    # Get total length of edges
    tot_meters = 0
    for i, (u, v, attr_dict) in enumerate(g_out.edges(data=True)):
        tot_meters += attr_dict['length']
    if verbose:
        print("Length of edges (km):", tot_meters / 1000)
    g_out.graph['Tot_edge_km'] = tot_meters / 1000
    if verbose:
        print("G.graph:", g_out.graph)
    # Save
    if len(g_out.nodes()) == 0:
        nx.write_gpickle(g_out, out_file, protocol=pickle_protocol)
        return
    # Save graph
    if verbose:
        logger.info("Saving graph to directory: {}".format(graph_dir))
    nx.write_gpickle(g_out, out_file, protocol=pickle_protocol)
    t7 = time.time()
    if verbose:
        print("Total time to run wkt_to_G():", t7 - t0, "seconds")
    return


class SpaceNet5WktToGraphTask(BaseTask):
    """
    Implements the functionality of step 05 in the CRESI framework.
    """
    schema = SpaceNet5WktToGraphTaskSchema

    def __init__(self, model, config):
        """
        Parameters
        ----------
            model : BaseModel
            config : Config
        """
        super().__init__(model, config)

    def run(self):
        """Implements the main logic behind this task"""
        simplify_graph = True
        verbose = True
        pickle_protocol = 4
        node_iter = 10000
        edge_iter = 10000  # start int for edge naming
        manually_reproject_nodes = False  # True
        n_threads = 12
        # Output files
        res_root_dir = os.path.join(self.config.path_results_root, self.config.test_results_dir)
        path_images = os.path.join(self.config.test_data_refined_dir)
        csv_file = os.path.join(res_root_dir, self.config.wkt_submission)
        graph_dir = os.path.join(res_root_dir, self.config.graph_dir)
        os.makedirs(graph_dir, exist_ok=True)
        min_subgraph_length_pix = self.config.min_subgraph_length_pix
        min_spur_length_m = self.config.min_spur_length_m
        # Read in wkt list
        logger.info("df_wkt at: {}".format(csv_file))
        df_wkt = pd.read_csv(csv_file)
        # Iterate through image ids and create graphs
        t0 = time.time()
        image_ids = np.sort(np.unique(df_wkt['ImageId']))
        n_files = len(image_ids)
        print("image_ids:", image_ids)
        print("len image_ids:", len(image_ids))
        n_threads = min(n_threads, n_files)
        params = []
        for i, image_id in enumerate(image_ids):
            out_file = os.path.join(graph_dir, image_id.split('.')[0] + '.gpickle')
            if verbose:
                logger.info("\n{x} / {y}, {z}".format(x=i + 1, y=len(image_ids), z=image_id))
            # For geo referencing, im_file should be the raw image
            if self.config.num_channels == 3:
                im_file = os.path.join(path_images, 'RGB-PanSharpen_' + image_id + '.tif')
            else:
                im_file = os.path.join(path_images, 'MUL-PanSharpen_' + image_id + '.tif')
            if not os.path.exists(im_file):
                im_file = os.path.join(path_images, image_id + '.tif')
            # Filter
            df_filter = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
            wkt_list = df_filter.values
            # Print a few values
            if verbose:
                logger.info("\n{x} / {y}, num linestrings: {z}".format(x=i + 1, y=len(image_ids), z=len(wkt_list)))
            if verbose:
                print("image_file:", im_file)
                print("  wkt_list[:2]", wkt_list[:2])
            if (len(wkt_list) == 0) or (wkt_list[0] == 'LINESTRING EMPTY'):
                g = nx.MultiDiGraph()
                nx.write_gpickle(g, out_file, protocol=pickle_protocol)
                continue
            else:
                params.append((wkt_list, im_file, min_subgraph_length_pix, node_iter, edge_iter, min_spur_length_m,
                               simplify_graph, self.config.rdp_epsilon, manually_reproject_nodes, out_file, graph_dir,
                               n_threads, verbose))
        # Execute
        if n_threads > 1:
            pool = Pool(n_threads)
            pool.map(wkt_to_g, params)
        else:
            wkt_to_g(params[0])
        tf = time.time()
        logger.info("Time to run wkt_to_G.py: {} seconds".format(tf - t0))
        print("Time to run wkt_to_G.py: {} seconds".format(tf - t0))
