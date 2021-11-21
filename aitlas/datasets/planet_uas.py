import csv
import matplotlib.pyplot as plt
import random

from itertools import compress
from .multilabel_classification import MultiLabelClassificationDataset

LABELS = ["haze", "primary", "agriculture", "clear", "water", "habitation", "road", "cultivation", "slash_burn",
          "cloudy", "partly_cloudy", "conventional_mine", "bare_ground", "artisinal_mine", "blooming",
          "selective_logging", "blow_down"]


class PlanetUASMultiLabelDataset(MultiLabelClassificationDataset):
    url = "https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/overview"

    labels = LABELS
    name = "Planet UAS multilabel dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        MultiLabelClassificationDataset.__init__(self, config)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # load image and remove last channel
        img = self.image_loader(self.data[index][0])[:, :, :3]
        if self.transform:
            img = self.transform(img)
        target = self.data[index][1]
        if self.target_transform:
            target = self.target_transform(self.data[index][1])
        return img, target

    def show_image(self, index):
        labels_list = list(compress(self.labels, self[index][1]))
        fig = plt.figure(figsize=(8, 6))
        plt.title(
            f"Image with index {index} from the dataset {self.get_name()}, with labels:\n "
            f"{str(labels_list).strip('[]')}\n",
            fontsize=14,
        )
        plt.axis("off")
        plt.imshow(self[index][0])
        return fig

    def show_batch(self, size):
        if size % 3:
            raise ValueError("The provided size should be divided by 3!")
        image_indices = random.sample(range(0, len(self.data)), size)
        figure_height = int(size / 3) * 4
        figure, ax = plt.subplots(int(size / 3), 3, figsize=(20, figure_height))
        figure.suptitle(
            "Example images with labels from {}".format(self.get_name()), fontsize=32, y=1.006
        )
        for axes, image_index in zip(ax.flatten(), image_indices):
            axes.imshow(self[image_index][0])
            labels_list = list(compress(self.labels, self[image_index][1]))
            str_label_list = ""
            if len(labels_list) > 4:
                str_label_list = f"{str(labels_list[0:4]).strip('[]')}\n"
                str_label_list += f"{str(labels_list[4:]).strip('[]')}\n"
            else:
                str_label_list = f"{str(labels_list).strip('[]')}\n"
            axes.set_title(str_label_list[:-1], fontsize=18, pad=10)
            axes.set_xticks([])
            axes.set_yticks([])
        figure.tight_layout()
        return figure


# Run this Function to convert the dataset in PASCAL VOC data format
def prepare(csv_train_file):
    f = open("multilabels.txt", "w")
    labels = []
    images = {}

    with open(csv_train_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                tmp_labels = row[1].split(' ')
                images[row[0]] = tmp_labels
                for label in tmp_labels:
                    if label not in labels:
                        labels.append(label)
                line_count += 1

    header = '\t'.join(labels)
    f.write("image\t" + header + "\n")

    for k, v in images.items():
        tmp_image = ""
        for label in labels:
            if label in v:
                tmp_image += "1\t"
            else:
                tmp_image += "0\t"
        f.write(k + "\t" + tmp_image[:-1] + "\n")
    f.close()


def kaggle_format(csv_file_path, output_file, threshold):
    f = open(csv_file_path, "w")
    labels = []
    images = {}

    with open(output_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                labels = row[1:]
                line_count += 1
            else:
                images[row[0]] = row[1:]
                line_count += 1
    header = "image_name" + "," + "tags"
    f.write(header + "\n")

    for k, v in images.items():
        tmp_image = ""
        for i, prob in enumerate(v):
            if float(prob) >= threshold:
                tmp_image += labels[i] + " "
        f.write(k.replace(".jpg", "") + "," + tmp_image[:-1] + "\n")
    f.close()
