import logging
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..base import BaseDataset
from .schemas import CropsDatasetSchema


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class CropsDataset(BaseDataset):
    """CropsDataset - a crop type classification dataset"""

    schema = CropsDatasetSchema

    def __init__(self, config):
        super().__init__(config)

    def preprocess(self):
        raise NotImplementedError(
            "Please implement the `preprocess` method for your crop type classification dataset"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (timeseries, target) where target is index of the target class.
        """
        row = self.index.iloc[index]

        h5path = self.h5path[row.region]
        if self.X_list is None:
            # Looks like this is what I need (load directly from file)
            with h5py.File(h5path, "r") as dataset:
                X = np.array(dataset[(row.path)])
        else:
            X = self.X_list[index]

        # translate CODE_CULTU to class id
        y = self.mapping.loc[row["CODE_CULTU"]].id

        if self.transform:
            X, y = self.transform((X, y))

        return X, y

    def get_labels(self):
        return self.index.classid

    def data_distribution_table(self):
        label_count = (
            self.index[["id", "region", "classname"]]
            .groupby(["classname", "region"])
            .count()
            .reset_index()
        )
        label_count.columns = ["Label", "Region", "Number of parcels"]
        return label_count

    def parcel_distribution_table(self):
        # Figure 2 a) in the paper
        parcel_count = (
            self.index[["id", "region"]].groupby("region").count().reset_index()
        )
        parcel_count.columns = ["Region NUTS-3", "# " + self.config.level]
        total_row = parcel_count.sum(axis=0)
        total_row["Region NUTS-3"] = "Total"

        parcel_count = parcel_count.append(total_row, ignore_index=True)
        return parcel_count

    def data_distribution_barchart(self):
        # Figure 2 b) in the paper
        label_count = self.data_distribution_table()
        fig, ax = plt.subplots(figsize=(12, 10))
        g = sns.barplot(
            x="Label", y="Number of parcels", hue="Region", data=label_count, ax=ax
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=30)
        g.set_yscale("log")
        return fig

    def show_samples(self):
        return self.index.head(20)

    def show_image(self, index):
        return self.show_timeseries(index)

    def show_timeseries(self, index):
        # Figure 3 in the paper
        X, _ = self.__getitem__(index)
        label = self.index.iloc[index].loc["classname"]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(
            f"Time series with index {index} from the region {self.index.iloc[index].loc['region']}, with label {label}\n",
            fontsize=14,
        )
        ax.plot(X)
        ax.legend(self.selected_bands[: X.shape[1]])
        ax.set_ylabel("ρ ")  # x ${10^4}$

        return fig

    def get_codes(self):
        return self.codes

    def load_classmapping(self, classmapping):
        if not os.path.exists(classmapping):
            if self.config.verbose:
                """
                TODO: either add a url for our dataset or remove it for breizhcrops
                """
        else:
            if self.config.verbose:
                logging.info(f"Found class mapping at {classmapping}")

        self.mapping = pd.read_csv(classmapping, index_col=0).sort_values(by="id")
        self.mapping = self.mapping.set_index("code")
        self.classes = self.mapping["id"].unique()
        self.classname = self.mapping.groupby("id").first().classname.values
        self.klassenname = self.classname
        self.nclasses = len(self.classes)
        if self.config.verbose:
            logging.info(f"read {self.nclasses} classes from {classmapping}")
