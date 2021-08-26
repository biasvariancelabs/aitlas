import csv
import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ..base import BaseDataset
from ..utils import image_loader
from .schemas import SegmentationDatasetSchema

#"Background": 0
#"Buildings": 1

LABELS = ["Background", "Buildings"]

"""
For the Inria Aerial Image Labeling dataset the mask is in one file, each label is color coded.
"""


class InriaDataset(BaseDataset):
    url = "https://project.inria.fr/aerialimagelabeling/"

    schema = SegmentationDatasetSchema
    labels = LABELS
    name = "Inria dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        BaseDataset.__init__(self, config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.root, self.config.csv_file_path)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index], True) / 255
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype('float32')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask

    def __len__(self):
        return len(self.images)

    def load_dataset(self, root_dir, file_path):
        if not self.labels:
            raise ValueError(
                "You need to provide the list of labels for the dataset"
            )
        with open(file_path, "r") as f:
            csv_reader = csv.reader(f)
            for index, row in enumerate(csv_reader):
                self.images.append(os.path.join(root_dir, row[0] + '.jpg'))
                self.masks.append(os.path.join(root_dir, row[0] + '_m.png'))

    def get_labels(self):
        return self.labels

    def show_image(self, index):
        legend_elements = [Patch(facecolor='white', label=self.labels[0]),
                           Patch(facecolor='black', label=self.labels[1])]

        img = self[index][0]
        mask = self[index][1]

        fig = plt.figure(figsize=(8, 6))
        fig.legend(handles=legend_elements)
        plt.title(f"Image and mask with index {index} from the dataset {self.get_name()}\n", fontsize=14)
        plt.subplot(1, 2, 1)
        plt.imshow(mask)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return fig


# Run this Function to split images into 500x500 pieces, and train file train.txt
# containing lists of pieces used for training
def prepare():
    IMGS_DIR = "./inria/images"
    MASKS_DIR = "./inria/masks"
    OUTPUT_DIR = "./inria/output"

    TARGET_SIZE = 500

    img_paths = glob.glob(os.path.join(IMGS_DIR, "*.tif"))
    mask_paths = glob.glob(os.path.join(MASKS_DIR, "*.tif"))
    file = open("train.txt", "w")

    img_paths.sort()
    mask_paths.sort()

    os.makedirs(OUTPUT_DIR)
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        print(img_path, mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    out_img_path = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                    file.write(img_filename + "\n")

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
        file.close()


