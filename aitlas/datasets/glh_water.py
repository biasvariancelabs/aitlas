import numpy as np
import os
import pandas as pd
import csv
from concurrent.futures import ThreadPoolExecutor

from .semantic_segmentation import SemanticSegmentationDataset
from ..utils import image_loader
from .schemas import FAIREOSchema

"""
The dataset aims at global surface water detection in very-high-resolution (VHR) satellite imagery. 
It consists of 250 satellite images and 40.96 billion pixels labeled surface water annotations that 
are distributed globally and contain water bodies exhibiting a wide variety of types. 
The initial size of the images is 12800 x 12800. This notebook divides the images into 1280 x 1280 
patches to increase the amount of trainning samples.
"""

# Define sliding window size
PATCH_H, PATCH_W = 1280, 1280

class GLHWaterDataset(SemanticSegmentationDataset):
    url = "https://drive.google.com/drive/folders/1VACDh3aGx72hDdz7Taf7FyUrpZS31Ksx"

    labels = ["non-water", "water"]
    color_mapping = [[0, 0, 0], [255, 255, 255]] 
    name = "GLH-Water"
    schema = FAIREOSchema

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.selection = self.config.selection
        self.csv_file = self.config.csv_file
        
        # Load the data during initialization
        self.images, self.masks = self.load_dataset(self.data_dir, self.csv_file)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        
        # Vectorized thresholding replaces the nested 1.6-million iteration loop.
        single_band_mask = np.where(mask == 255, 1, 0).astype(np.uint8)

        # Create one-hot encoded mask stack
        masks = [(single_band_mask == v) for v in range(len(self.labels))]
        mask = np.stack(masks, axis=-1).astype("float32")

        return self.apply_transformations(image, mask)

    def load_dataset(self, data_dir, csv_file):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        self.images = []
        self.masks = []

        ids = set(os.listdir(os.path.join(data_dir, "img")))

        # new 60/20/20 splits are provided by BVLabs
        if csv_file:
            ids = set()
            with open(csv_file, mode='r') as file:
                csvFile = csv.reader(file)
                for lines in csvFile:
                    if lines:
                        ids.add(lines[0].split('/')[1])

        self.images_path = [os.path.join(data_dir, "img", image_id) for image_id in ids]
        self.masks_path = [os.path.join(data_dir, "label", image_id[: image_id.rfind('.jpg')]+'.png') for image_id in ids]

        # --- MULTI-THREADING SETUP ---
        def process_single_image(img_path, mask_path):
            """Helper function to load and slice a single image on a separate thread."""
            image = image_loader(img_path) 
            mask = image_loader(mask_path)
            
            img_patches = []
            mask_patches = []
            stride = 1280
            
            img_h, img_w = image.shape[:2]
            
            for y in range(0, img_h - PATCH_H + 1, stride):
                for x in range(0, img_w - PATCH_W + 1, stride):
                    img_patches.append(image[y:y + PATCH_H, x:x + PATCH_W])
                    mask_patches.append(mask[y:y + PATCH_H, x:x + PATCH_W])
                    
            return img_patches, mask_patches

        print(f"Loading {len(self.images_path)} images into RAM using multi-threading...")
        
        # ThreadPoolExecutor reads multiple files from the disk concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Map the helper function to all image and mask paths simultaneously
            results = executor.map(process_single_image, self.images_path, self.masks_path)
            
            # As threads finish, extend the main lists
            for img_patches, mask_patches in results:
                self.images.extend(img_patches)
                self.masks.extend(mask_patches)

        print("Finished loading dataset into memory!")
        return self.images, self.masks

    def data_distribution_table(self):
        label_dist = {key: 0 for key in self.labels}
        for image, mask in self.dataloader():
            for index, label in enumerate(self.labels):
                label_dist[self.labels[index]] += mask[:, :, :, index].sum()
        label_count = pd.DataFrame.from_dict(label_dist, orient='index')
        label_count.columns = ["Number of pixels"]
        label_count = label_count.astype(float)
        return label_count