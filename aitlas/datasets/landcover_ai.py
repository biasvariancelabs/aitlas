import glob
import os
import cv2
import numpy as np

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset

"""
41 orthophoto tiles from different counties located in all regions. Every tile has about 5 km2.
There are 33 images with resolution 25cm (ca. 9000 × 9500 px) and 8 images with resolution 50cm (ca. 4200 × 4700 px)
Tne masks are codded with building (1), woodland (2), water (3), and road (4)
Use function split_images to split the images and the masks in smaller patches
"""


class LandCoverAiDataset(SemanticSegmentationDataset):
    url = "https://landcover.ai.linuxpolska.com/"

    labels = ["Background", "Buildings", "Woodlands", "Water", "Road"]
    color_mapping = [[255, 255, 0], [0, 0, 0], [0, 255, 0], [0, 0, 255], [200, 200, 200]]
    name = "Landcover AI"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])[:, :, 1]
        # extract certain classes from mask (e.g. Buildings)
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)


def split_images(imgs_dir, masks_dir, output_dir):
    target_size = 512

    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))

    img_paths.sort()
    mask_paths.sort()

    os.makedirs(output_dir)
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], target_size):
            for x in range(0, img.shape[1], target_size):
                img_tile = img[y: y + target_size, x: x + target_size]
                mask_tile = mask[y: y + target_size, x: x + target_size]

                if (
                    img_tile.shape[0] == target_size
                    and img_tile.shape[1] == target_size
                ):
                    out_img_path = os.path.join(
                        output_dir, "{}_{}.jpg".format(img_filename, k)
                    )
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(
                        output_dir, "{}_{}_m.png".format(mask_filename, k)
                    )
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
