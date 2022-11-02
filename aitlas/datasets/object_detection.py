import os
import torch
from xml.etree import ElementTree as et
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..base import BaseDataset
from ..utils import image_loader, collate_fn
from .schemas import ObjectDetectionDatasetSchema

"""
Generic dataset for the task of semantic segmentation
"""


class ObjectDetectionDataset(BaseDataset):
    schema = ObjectDetectionDatasetSchema

    # labels: 0 index is reserved for background
    labels = [None, 'apple', 'banana', 'orange']
    name = None

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.data_dir = self.config.data_dir
        self.images = []
        self.load_dataset(self.data_dir)

    def __getitem__(self, index):
        img_name = self.images[index]
        image = image_loader(os.path.join(self.data_dir, img_name)) / 255.0

        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.data_dir, annot_filename)
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # box coordinates for xml files are extracted
        for member in root.findall('object'):
            labels.append(self.labels.index(member.find('name').text))

            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        # image_id
        image_id = torch.tensor([index])
        target["image_id"] = image_id

        return self.apply_transformations(image, target)

    def dataloader(self):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
            # drop_last=True,
        )

    def __len__(self):
        return len(self.images)

    def apply_transformations(self, image, target):
        if self.joint_transform:
            image, target = self.joint_transform((image, target))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def load_dataset(self, data_dir):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")
        self.images = [image for image in sorted(os.listdir(data_dir)) if image[-4:] == '.jpg']

    def get_labels(self):
        return self.labels

    def data_distribution_table(self):
        pass

    def data_distribution_barchart(self, show_title=True):
        pass

    def show_image(self, index, show_title=False):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        img, target = self[index]
        fig, a = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)
        a.imshow(img)
        for box, label in zip(target['boxes'], target['labels']):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
            a.annotate(self.labels[label], (box[0], box[1]), color='black', weight='bold', fontsize=12, ha='center', va='center')
        plt.show()
        return fig

