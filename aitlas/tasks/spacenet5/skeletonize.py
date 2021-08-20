"""
Notes
-----
    Based on the implementation at:
        https://github.com/CosmiQ/cresi/blob/master/cresi/04_skeletonize.py
"""
import logging
import os
import time
from collections import OrderedDict
from itertools import tee
from multiprocessing.pool import Pool

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import skimage
import skimage.draw
import skimage.io
from matplotlib.pylab import plt
from numba import jit
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

from ...base import BaseTask, Config
from ..schemas import SpaceNet5SkeletonizeTaskSchema


# Create or get the logger
logger = logging.getLogger(__name__)
# set log level
logger.setLevel(logging.INFO)
linestring = "LINESTRING {}"


def neighbors(shape):
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])


@jit(nopython=True)
def mark(img, nbs):  # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
        s = 0
        for dp in nbs:
            if img[p + dp] != 0:
                s += 1
        if s == 2:
            img[p] = 1
        else:
            img[p] = 2


@jit(nopython=True)
def idx2rc(idx, acc):
    """
    Trans index to r, c...
    """
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst


@jit(nopython=True)
def fill(img, p, num, nbs, acc, buf):
    """
    Fill a node (may be two or more points).
    """
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0
    s = 1
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == back:
                img[cp] = num
                buf[s] = cp
                s += 1
        cur += 1
        if cur == s:
            break
    return idx2rc(buf[:s], acc)


@jit(nopython=True)
def trace(img, p, nbs, acc, buf):
    """
    Trace the edge and use a buffer, then buf.copy, if use [] numba not works.
    """
    c1 = 0
    c2 = 0
    newp = 0
    cur = 0
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1 == 0:
                    c1 = img[cp]
                else:
                    c2 = img[cp]
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2 != 0:
            break
    return c1 - 10, c2 - 10, idx2rc(buf[:cur], acc)


@jit(nopython=True)
def parse_structure(img, pts, nbs, acc):
    """
    Parse the image then get the nodes and edges.
    """
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)
    edges = []
    for p in pts:
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges


def build_graph(nodes, edges, multi=False):
    """
    Use nodes and edges build a networkx graph.
    """
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s, e, pts in edges:
        ll = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=ll)
    return graph


def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape) + 2), dtype=np.uint16)
    buf[tuple([slice(1, -1)] * buf.ndim)] = ske
    return buf


def build_sknw(ske, multi=False):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,) + buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    pts = np.array(np.where(buf.ravel() == 2))[0]
    nodes, edges = parse_structure(buf, pts, nbs, acc)
    return build_graph(nodes, edges, multi)


def draw_graph(img, graph, cn=255, ce=128):
    """
    Draw the graph.
    """
    acc = np.cumprod((1,) + img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for idx in graph.nodes():
        pts = graph.node[idx]["pts"]
        img[np.dot(pts, acc)] = cn
    for (s, e) in graph.edges():
        eds = graph[s][e]
        for i in eds:
            pts = eds[i]["pts"]
            img[np.dot(pts, acc)] = ce


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def remove_sequential_duplicates(seq):
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res


def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx + 1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx - 1] : v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """
    Compute angle (in degrees) for p0p1p2 corner

    Parameters
    ----------
            p0 , p1, p2
                points in the form of [x, y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def preprocess(
    image,
    thresh,
    image_multiplier=255,
    hole_size=300,
    cv2_kernel_close=7,
    cv2_kernel_open=7,
    verbose=False,
):
    """
    http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_holes
    hole_size in remove_small_objects is the maximum area, in pixels of the hole
    """
    # Sometimes get a memory error with this approach
    if image.size < 10000000000:
        if verbose:
            print("Run preprocess() with skimage")
        image = (image > (image_multiplier * thresh)).astype(np.bool)
        remove_small_objects(image, hole_size, in_place=True)
        remove_small_holes(image, hole_size, in_place=True)
    # cv2 is generally far faster and more memory efficient (though less effective)
    else:
        if verbose:
            print("Run preprocess() with cv2")
        kernel_close = np.ones((cv2_kernel_close, cv2_kernel_close), np.uint8)
        kernel_open = np.ones((cv2_kernel_open, cv2_kernel_open), np.uint8)
        kernel_blur = cv2_kernel_close
        # Global thresh
        blur = cv2.medianBlur((image * image_multiplier).astype(np.uint8), kernel_blur)
        glob_thresh_arr = cv2.threshold(blur, thresh, 1, cv2.THRESH_BINARY)[1]
        glob_thresh_arr_smooth = cv2.medianBlur(glob_thresh_arr, kernel_blur)
        mask_thresh = glob_thresh_arr_smooth
        # Opening and closing
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        closing_t = cv2.morphologyEx(mask_thresh, cv2.MORPH_CLOSE, kernel_close)
        opening_t = cv2.morphologyEx(closing_t, cv2.MORPH_OPEN, kernel_open)
        image = opening_t.astype(np.bool)
    return image


def graph2lines(g):
    node_lines = []
    edges = list(g.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def visualize(image, g, vertices):
    plt.imshow(image, cmap="gray")
    # Draw edges by pts
    for (s, e) in g.edges():
        values = flatten([[v] for v in g[s][e].values()])
        for val in values:
            ps = val.get("pts", [])
            plt.plot(ps[:, 1], ps[:, 0], "green")
    ps = np.array(vertices)
    plt.plot(ps[:, 1], ps[:, 0], "r.")
    plt.title("Build Graph")
    plt.show()


def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(
        line1[1] - line1[0]
    )


def remove_small_terminal(
    g, weight="weight", min_weight_val=30, pix_extent=1300, edge_buffer=4, verbose=False
):
    """
    Remove small terminals, if a node in the terminal is within edge_buffer of the the graph edge, keep it
    """
    deg = dict(g.degree())
    terminal_points = [i for i, d in deg.items() if d == 1]
    if verbose:
        print("remove_small_terminal() - N terminal_points:", len(terminal_points))
    edges = list(g.edges())
    for s, e in edges:
        if s == e:
            sum_len = 0
            values = flatten([[v] for v in g[s][s].values()])
            for ix, val in enumerate(values):
                sum_len += len(val["pts"])
            if sum_len < 3:
                g.remove_edge(s, e)
                continue
        # Check if at edge
        sx, sy = g.nodes[s]["o"]
        ex, ey = g.nodes[e]["o"]
        edge_point = False
        for p_tmp in [sx, sy, ex, ey]:
            if (p_tmp < (0 + edge_buffer)) or (p_tmp > (pix_extent - edge_buffer)):
                if verbose:
                    print("p_tmp:", p_tmp)
                    print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                    print(
                        "(p_tmp > (pix_extent - edge_buffer):",
                        (p_tmp > (pix_extent - edge_buffer)),
                    )
                    print("p_tmp < (0 + edge_buffer):", (p_tmp < (0 + edge_buffer)))
                edge_point = True
            else:
                continue
        # Don't remove edges near the edge of the image
        if edge_point:
            if verbose:
                print("(pix_extent - edge_buffer):", (pix_extent - edge_buffer))
                print("edge_point:", sx, sy, ex, ey, "continue")
            continue
        values = flatten([[v] for v in g[s][e].values()])
        for ix, val in enumerate(values):
            if verbose:
                print("val.get(weight, 0):", val.get(weight, 0))
            if s in terminal_points and val.get(weight, 0) < min_weight_val:
                g.remove_node(s)
            if e in terminal_points and val.get(weight, 0) < min_weight_val:
                g.remove_node(e)
    return


def add_direction_change_nodes(pts, s_coord, e_coord):
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


def add_small_segments(
    g,
    terminal_points,
    terminal_lines,
    dist1=24,
    dist2=80,
    angle1=30,
    angle2=150,
    verbose=False,
):
    """
    Connect small, missing segments terminal points are the end of edges.
    This function tries to pair small gaps in roads.
    It will not try to connect a missed T-junction, as the crossroad will not have a terminal point
    """
    print("Running add_small_segments()")
    try:
        node = g.node
    except:
        node = g.nodes
    term = [node[t]["o"] for t in terminal_points]
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if g.has_edge(s, e):
            continue
        good_pairs.append((s, e))
    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if g.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])
        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1 * angle1 < angle < angle1) or (angle < -1 * angle2) or (angle > angle2):
            good_pairs.append((s, e))
    if verbose:
        print("  good_pairs:", good_pairs)
    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [g.nodes[s]["o"], g.nodes[e]["o"]]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)
    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))
    wkt = []
    added = set()
    good_coordinates = []
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = (
                g.nodes[s]["o"].astype(np.int32),
                g.nodes[e]["o"].astype(np.int32),
            )
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = "(" + ", ".join(line_strings) + ")"
            wkt.append(linestring.format(line))
            good_coordinates.append((tuple(s_d), tuple(e_d)))
    return wkt, good_pairs, good_coordinates


def make_skeleton(
    image_location,
    thresh,
    fix_borders,
    replicate=5,
    clip=2,
    img_shape=(1300, 1300),
    image_multiplier=255,
    hole_size=300,
    cv2_kernel_close=7,
    cv2_kernel_open=7,
    use_medial_axis=False,
    max_out_size=(200000, 200000),
    num_classes=1,
    skeleton_band="all",
    verbose=False,
):
    """
    Extract a skeleton from a mask.
    skeleton_band is the index of the band of the mask to use for skeleton extraction,
    set to string 'all' to use all bands
    """
    if verbose:
        print("Executing make_skeleton...")
    t0 = time.time()
    rec = replicate + clip
    # Read in data
    if num_classes == 1:
        try:
            img = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
        except:
            img = skimage.io.imread(image_location, as_gray=True).astype(np.uint8)
    else:
        img_tmp = skimage.io.imread(image_location).astype(np.uint8)
        # We want skimage to read in (channels, h, w) for multi-channel
        # Assume less than 20 channels
        if img_tmp.shape[0] > 20:
            img_full = np.moveaxis(img_tmp, 0, -1)
        else:
            img_full = img_tmp
        # Select the desired band for skeleton extraction
        # If < 0, sum all bands
        if type(skeleton_band) == str:
            img = np.sum(img_full, axis=0).astype(np.int8)
        else:
            img = img_full[skeleton_band, :, :]
    if verbose:
        print("make_skeleton(), input img_shape:", img_shape)
        print("make_skeleton(), img.shape:", img.shape)
        print("make_skeleton(), img.size:", img.size)
        print("make_skeleton(), img dtype:", img.dtype)
    # Potentially keep only subset of data
    shape0 = img.shape
    img = img[: max_out_size[0], : max_out_size[1]]
    if img.shape != shape0:
        print("Using only subset of data!!!!!!!!")
        print("make_skeleton() new img.shape:", img.shape)
    if fix_borders:
        img = cv2.copyMakeBorder(
            img, replicate, replicate, replicate, replicate, cv2.BORDER_REPLICATE
        )
    if verbose:
        print("Run preprocess()...")
    t1 = time.time()
    img = preprocess(
        img,
        thresh,
        image_multiplier=image_multiplier,
        hole_size=hole_size,
        cv2_kernel_close=cv2_kernel_close,
        cv2_kernel_open=cv2_kernel_open,
    )
    t2 = time.time()
    if verbose:
        print("Time to run preprocess():", t2 - t1, "seconds")
    if not np.any(img):
        return None, None
    if not use_medial_axis:
        if verbose:
            print("skeletonize...")
        ske = skeletonize(img).astype(np.uint16)
        t3 = time.time()
        if verbose:
            print("Time to run skimage.skeletonize():", t3 - t2, "seconds")
    else:
        if verbose:
            print("running updated skimage.medial_axis...")
        ske = skimage.morphology.medial_axis(img).astype(np.uint16)
        t3 = time.time()
        if verbose:
            print("Time to run skimage.medial_axis():", t3 - t2, "seconds")
    if fix_borders:
        if verbose:
            print("fix_borders...")
        ske = ske[rec:-rec, rec:-rec]
        ske = cv2.copyMakeBorder(
            ske, clip, clip, clip, clip, cv2.BORDER_CONSTANT, value=0
        )
        img = img[replicate:-replicate, replicate:-replicate]
        t4 = time.time()
        if verbose:
            print("Time fix borders:", t4 - t3, "seconds")
    t1 = time.time()
    if verbose:
        print("ske.shape:", ske.shape)
        print("Time to run make_skeleton:", t1 - t0, "seconds")
    return img, ske


def image_to_skeleton_graph(params):
    (
        img_loc,
        out_ske_file,
        out_g_pickle,
        thresh,
        debug,
        fix_borders,
        img_shape,
        skeleteon_replicate,
        skeleton_clip,
        image_multiplier,
        hole_size,
        cv2_kernel_close,
        cv2_kernel_open,
        min_subgraph_length_pix,
        min_spur_length_pix,
        max_out_size,
        use_medial_axis,
        num_classes,
        skeleton_band,
        kernel_blur,
        min_background_frac,
        verbose,
    ) = params
    # Create skeleton
    img_refine, ske = make_skeleton(
        image_location=img_loc,
        thresh=thresh,
        fix_borders=fix_borders,
        replicate=skeleteon_replicate,
        clip=skeleton_clip,
        img_shape=img_shape,
        image_multiplier=image_multiplier,
        hole_size=hole_size,
        cv2_kernel_close=cv2_kernel_close,
        cv2_kernel_open=cv2_kernel_open,
        max_out_size=max_out_size,
        skeleton_band=skeleton_band,
        num_classes=num_classes,
        use_medial_axis=use_medial_axis,
        verbose=verbose,
    )

    if ske is None:
        return [linestring.format("EMPTY"), [], []]
    # Save to file
    if out_ske_file:
        cv2.imwrite(out_ske_file, ske.astype(np.uint8) * 255)
    # Create graph
    if verbose:
        print("Execute sknw...")
    # If the file is too large, use sknw_int64 to accommodate high numbers for coordinates
    if np.max(ske.shape) > 32767:
        g = build_sknw(ske, multi=True)
    else:
        g = build_sknw(ske, multi=True)
    # Iteratively clean out small terminals
    for i_tmp in range(8):
        n_tmp0 = len(g.nodes())
        if verbose:
            print("Clean out small terminals - round", i_tmp)
            print("Clean out small terminals - round", i_tmp, "num nodes:", n_tmp0)
        # sknw attaches a 'weight' property that is the length in pixels
        pix_extent = np.max(ske.shape)
        remove_small_terminal(
            g,
            weight="weight",
            min_weight_val=min_spur_length_pix,
            pix_extent=pix_extent,
        )
        # Kill the loop if we stopped removing nodes
        n_tmp1 = len(g.nodes())
        if n_tmp0 == n_tmp1:
            break
        else:
            continue
    if verbose:
        print("len G.nodes():", len(g.nodes()))
        print("len G.edges():", len(g.edges()))
    if len(g.edges()) == 0:
        return [linestring.format("EMPTY"), [], []]
    # Remove self loops
    e_bunch = nx.selfloop_edges(g)
    g.remove_edges_from(list(e_bunch))
    # Save G
    if len(out_g_pickle) > 0:
        nx.write_gpickle(g, out_g_pickle)
    return g, ske, img_refine


def g_to_wkt(g, add_small=True, img_copy=None, debug=False, verbose=False):
    """
    Transform G to wkt
    """
    if g == [linestring.format("EMPTY")] or type(g) == str:
        return [linestring.format("EMPTY")]
    node_lines = graph2lines(g)
    if not node_lines:
        return [linestring.format("EMPTY")]
    try:
        node = g.node
    except:
        node = g.nodes
    deg = dict(g.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]
    # Refine wkt
    if verbose:
        print("Refine wkt...")
    terminal_lines = {}
    vertices = []
    for i, w in enumerate(node_lines):
        if ((i % 10000) == 0) and (i > 0) and verbose:
            print("  ", i, "/", len(node_lines))
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            values = flatten([[v] for v in g[s][e].values()])
            for ix, val in enumerate(values):
                s_coord, e_coord = node[s]["o"], node[e]["o"]
                pts = val.get("pts", [])
                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)
                ps = add_direction_change_nodes(pts, s_coord, e_coord)
                if len(ps.shape) < 2 or len(ps) < 2:
                    continue
                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue
                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)
                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        for coord_list in segments:
            if len(coord_list) > 1:
                line = "(" + ", ".join(coord_list) + ")"
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format("(" + line + ")"))
    if add_small and len(terminal_points) > 1:
        small_segments, good_pairs, good_coordinates = add_small_segments(
            g, terminal_points, terminal_lines, verbose=verbose
        )
        print("small_segments", small_segments)
        wkt.extend(small_segments)
    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, g, vertices)
    if not wkt:
        return [linestring.format("EMPTY")]
    return wkt


def build_wkt_dir(
    in_directory,
    outfile,
    out_ske_dir,
    out_g_dir="",
    thresh=0.3,
    debug=False,
    add_small=True,
    fix_borders=True,
    image_shape=(1300, 1300),
    skeleton_replicate=5,
    skeleton_clip=2,
    image_multiplier=255,
    hole_size=300,
    cv2_kernel_close=7,
    cv2_kernel_open=7,
    min_subgraph_length_pix=50,
    min_spur_length_pix=16,
    space_net_naming_convention=False,
    num_classes=1,
    max_out_size=(100000, 100000),
    use_medial_axis=True,
    skeleton_band="all",
    kernel_blur=27,
    min_background_fraction=0.2,
    n_threads=12,
    verbose=False,
):
    """
    Execute built_graph_wkt for an entire folder.
    """
    image_files = np.sort([z for z in os.listdir(in_directory) if z.endswith(".tif")])
    n_files = len(image_files)
    n_threads = min(n_threads, n_files)
    params = []
    for i, image_file in enumerate(image_files):
        if verbose:
            print("\n", i + 1, "/", n_files, ":", image_file)
        logger.info("{x} / {y} : {z}".format(x=i + 1, y=n_files, z=image_file))
        img_loc = os.path.join(in_directory, image_file)
        if space_net_naming_convention:
            im_root = "AOI" + image_file.split("AOI")[-1].split(".")[0]
        else:
            im_root = image_file.split(".")[0]
        if verbose:
            print("  img_loc:", img_loc)
            print("  im_root:", im_root)
        if out_ske_dir:
            out_ske_file = os.path.join(out_ske_dir, image_file)
        else:
            out_ske_file = ""
        if verbose:
            print("  out_ske_file:", out_ske_file)
        if len(out_g_dir) > 0:
            out_g_pickle = os.path.join(
                out_g_dir, image_file.split(".")[0] + ".gpickle"
            )
        else:
            out_g_pickle = ""
        param_row = (
            img_loc,
            out_ske_file,
            out_g_pickle,
            thresh,
            debug,
            fix_borders,
            image_shape,
            skeleton_replicate,
            skeleton_clip,
            image_multiplier,
            hole_size,
            cv2_kernel_close,
            cv2_kernel_open,
            min_subgraph_length_pix,
            min_spur_length_pix,
            max_out_size,
            use_medial_axis,
            num_classes,
            skeleton_band,
            kernel_blur,
            min_background_fraction,
            verbose,
        )
        params.append(param_row)
    # Execute
    if n_threads > 1:
        pool = Pool(n_threads)
        pool.map(image_to_skeleton_graph, params)
    else:
        image_to_skeleton_graph(params[0])
    # Now build wkt_list (single-threaded)
    all_data = []
    for g_pickle in os.listdir(out_g_dir):
        t1 = time.time()
        gpath = os.path.join(out_g_dir, g_pickle)
        image_file = g_pickle.split(".")[0] + ".tif"
        if space_net_naming_convention:
            im_root = "AOI" + image_file.split("AOI")[-1].split(".")[0]
        else:
            im_root = image_file.split(".")[0]
        g = nx.read_gpickle(gpath)
        wkt_list = g_to_wkt(g, add_small=add_small, verbose=verbose)
        # Add to all_data
        for v in wkt_list:
            all_data.append((im_root, v))
        t2 = time.time()
        logger.info("Time to build graph: {} seconds".format(t2 - t1))
    # Save to csv
    df = pd.DataFrame(all_data, columns=["ImageId", "WKT_Pix"])
    df.to_csv(outfile, index=False)
    return df


class SpaceNet5SkeletonizeTask(BaseTask):
    """
    Implements the functionality of step 04 in the CRESI framework.
    """

    schema = SpaceNet5SkeletonizeTaskSchema  # set up the schema for the task

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
        """
        # Initialize some parameters
        add_small = True
        verbose = True
        super_verbose = False
        space_net_naming_convention = False
        debug = False
        fix_borders = True
        img_shape = ()
        skeleton_replicate = 5
        skeleton_clip = 2
        image_multiplier = 255
        hole_size = 300
        cv2_kernel_close = 7
        cv2_kernel_open = 7
        kernel_blur = -1
        min_background_fraction = -1
        max_out_size = (2000000, 2000000)
        n_threads = 12
        min_spur_length_pix = int(
            np.rint(self.config.min_spur_length_m / self.config.GSD)
        )
        print("min_spur_length_pix:", min_spur_length_pix)
        use_medial_axis = bool(self.config.use_medial_axis)
        print("Use_medial_axis?", use_medial_axis)
        # Output files
        res_root_dir = os.path.join(
            self.config.path_results_root, self.config.test_results_dir
        )
        outfile_csv = os.path.join(res_root_dir, self.config.wkt_submission)
        out_ske_dir = os.path.join(
            res_root_dir, self.config.skeleton_dir
        )  # set to '' to not save
        os.makedirs(out_ske_dir, exist_ok=True)
        if len(self.config.skeleton_pkl_dir) > 0:
            out_g_dir = os.path.join(
                res_root_dir, self.config.skeleton_pkl_dir
            )  # set to '' to not save
            os.makedirs(out_g_dir, exist_ok=True)
        else:
            out_g_dir = ""
        print("masks_dir:", self.config.masks_dir)
        print("out_ske_dir:", out_ske_dir)
        print("out_g_dir:", out_g_dir)
        thresh = self.config.skeleton_thresh
        min_subgraph_length_pix = self.config.min_subgraph_length_pix
        t0 = time.time()
        df = build_wkt_dir(
            self.config.masks_dir,
            outfile_csv,
            out_ske_dir,
            out_g_dir,
            thresh,
            debug=debug,
            add_small=add_small,
            fix_borders=fix_borders,
            image_shape=img_shape,
            skeleton_replicate=skeleton_replicate,
            skeleton_clip=skeleton_clip,
            image_multiplier=image_multiplier,
            hole_size=hole_size,
            min_subgraph_length_pix=min_subgraph_length_pix,
            min_spur_length_pix=min_spur_length_pix,
            cv2_kernel_close=cv2_kernel_close,
            cv2_kernel_open=cv2_kernel_open,
            max_out_size=max_out_size,
            skeleton_band=self.config.skeleton_band,
            num_classes=self.config.num_classes,
            space_net_naming_convention=space_net_naming_convention,
            use_medial_axis=use_medial_axis,
            kernel_blur=kernel_blur,
            n_threads=n_threads,
            min_background_fraction=min_background_fraction,
            verbose=verbose,
        )
        print("len df:", len(df))
        print("outfile:", outfile_csv)
        t1 = time.time()
        logger.info("Total time to run build_wkt_dir: {} seconds".format(t1 - t0))
        print("Total time to run build_wkt_dir:", t1 - t0, "seconds")
