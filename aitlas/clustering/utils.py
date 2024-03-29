import logging
import time

import faiss
import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.

    :param npdata: features to preprocess
    :type npdata: np.array (N * dim)
    :param pca: dim of output
    :type pca: int
    :return: data PCA-reduced, whitened and L2-normalized
    :rtype: np.array (N * pca)
    """
    _, ndim = npdata.shape
    npdata = npdata.astype("float32")

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.

    :param xb: data
    :type xb: np.array (N * dim)
    :param nnn: number of nearest neighbors
    :type nnn: int
    :return: list for each data the list of ids to its nnn nearest neighbors
    :return: list for each data the list of distances to its nnn NN
    :rtype: np.array (N * nnn)
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.

    :param image_indexes: list of data indexes
    :type image_indexes: list of ints
    :param pseudolabels: list of labels for each data
    :type pseudolabels: list of ints
    :param dataset: initial dataset
    :type dataset: list of tuples with paths to images
    :param transform: a function/transform that takes in an PIL image and returns a transformed version
    :type transform: callable, optional
    """

    def __init__(self, image_indexes, pseudolabels, dataset):
        self.pseudolabels = self.make_dataset(image_indexes, pseudolabels)
        self.dataset = dataset

    def make_dataset(self, image_indexes, pseudolabels):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        pseudolabels = []
        for j, idx in enumerate(image_indexes):
            pseudolabels.append(label_to_idx[pseudolabels[j]])
        return pseudolabels

    def __getitem__(self, index):
        """
        :params index: index of data
        :type index: int
        :return: tuple (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        return self.dataset.__getitem__(index)[0], self.pseudolabels[index]

    def __len__(self):
        return len(self.pseudolabels)


def cluster_assign(images_lists, dataset):
    """Creates a dataset from clustering, with clusters as labels.

    :params images_lists: for each cluster, the list of image indexes belonging to this cluster
    :type images_lists: list of lists of ints
    :params dataset: initial dataset
    :type dataset: list of tuples with paths to images
    :return: dataset with clusters as labels
    :rtype: ReassignedDataset(torch.utils.data.Dataset)
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    return ReassignedDataset(image_indexes, pseudolabels, dataset)


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    :param x: data
    :type x: np.array (N * dim)
    :param nmb_clusters: number of clusters
    :type nmb_clusters: int
    :return: list of ids for each data to its nearest cluster
    :rtype: list of ints
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Kmeans(d, nmb_clusters)
    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x)
    dists, I = clus.index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        logging.info("k-means loss evolution: {0}".format(losses))

    return [int(n[0]) for n in I], losses[-1]


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.

    :param I: for each vertex the ids to its nnn linked vertices + first column of identity.
    :type I: numpy array
    :param D: for each data the l2 distances to its nnn linked vertices + first column of zeros.
    :type D: numpy array
    :param sigma: bandwith of the Gaussian kernel.
    :type sigma: float
    :return:  affinity matrix of the graph.
    :rtype: scipy.sparse.csr_matrix
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype("float32")

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype="float32")

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert n == m
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert assign[i] >= 0
    return assign
