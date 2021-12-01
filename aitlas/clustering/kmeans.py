import logging
import time

from PIL import ImageFile

from .utils import preprocess_features, run_kmeans


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


ImageFile.LOAD_TRUNCATED_IMAGES = True


class Kmeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        start = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, verbose)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            logging.info("k-means time: {0:.0f} s".format(time.time() - start))

        return loss
