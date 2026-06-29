from .multiclass_classification import MultiClassClassificationDataset
from PIL import Image
import pandas as pd
import numpy as np
import io
import os

'''
RSC11 Dataset is a remote sensing dataset designed for land cover classification using 
high-resolution satellite imagery. It contains 11 classes representing different land cover types. 
The dataset includes 1232 images in total, with each class about 100 images 
'''

LABELS = ["dense_forest", "grassland", "harbor", "high_buildings", "low_buildings",
          "overpass", "railway", "residential_area", "roads", "sparse_forest", "storage_tanks"]

class RSC11Dataset(MultiClassClassificationDataset):
    url = "https://huggingface.co/datasets/jonathan-roberts1/RS_C11"
    labels = LABELS
    name = "RSC11 dataset"

    def __init__(self, config):
        # now call the constructor to validate the schema and load the data
        super().__init__(config)
        # load the data
        self.data_dir = self.config.data_dir
        self.csv_file = self.config.csv_file
        self.data = self.load_dataset()

    def __getitem__(self, index):
        """
        :param index: Index
        :type index: int
        :return: tuple where target is index of the target class.
        :rtype: tuple (image, target)

        """    
        # get raw bytes and label index tuple
        img_bytes, target = self.data[index]
        # load image from bytes directly
        img = np.asarray(Image.open(io.BytesIO(img_bytes)))
        # apply transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        
        return img, target    
    
    def load_dataset(self):
        files = sorted(os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir))

        if not files:
            raise FileNotFoundError(f"No files found in directory: {self.data_dir}")

        # Load and combine all parquet files into a single DataFrame
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

        if self.csv_file: # For splits, csv file contains indexes
            idx_df = pd.read_csv(self.csv_file, header=None, names=['index', 'label'])
            # Filter parquet df to only rows with those indexes
            df = df.iloc[idx_df['index'].values]

        data = []
        for _, row in df.iterrows():
            img_bytes = row['image']['bytes']  # raw image bytes
            label = row['label']
            data.append((img_bytes, label))
        
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        
        return data
    
    def show_samples(self):
        df = pd.DataFrame(self.data, columns=["Image", "LabelIndex"])

        # Map label indices to label names
        df["Label"] = df["LabelIndex"].apply(lambda x: self.labels[x])

        # Return a subset with raw image bytes and label name
        return df[["Image", "Label"]].head(20)


    def data_distribution_table(self):
        labels = [label for _, label in self.data]
        df = pd.DataFrame(labels, columns=['LabelIndex'])

        # Count labels
        label_counts = df['LabelIndex'].value_counts().reset_index()
        label_counts.columns = ['LabelIndex', 'Count']
        label_counts['Label'] = label_counts['LabelIndex'].apply(lambda x: self.labels[x])

        return label_counts[['Label', 'Count']]