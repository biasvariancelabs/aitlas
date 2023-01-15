import csv
import importlib
import os
from time import time
import glob
import cv2
import numpy as np
import tifffile
import torch
import subprocess

from PIL import Image, ImageOps


def get_class(class_name):
    """returns the class type for a given class name. Expects a string of type `module.submodule.Class`"""
    module = class_name[: class_name.rindex(".")]
    cls = class_name[class_name.rindex(".") + 1:]
    return getattr(importlib.import_module(module), cls)


def current_ts():
    """returns current timestamp in secs"""
    return int(time())


def pil_loader(file, convert_to_grayscale=False):
    """open an image from disk"""
    with open(file, "rb") as f:
        if convert_to_grayscale:
            return np.asarray(Image.open(f).convert("L"))
        return np.asarray(Image.open(f))


def tiff_loader(file):
    """opens a tiff image from disk"""
    return tifffile.imread(file)


def image_loader(file_path, convert_to_grayscale=False):
    filename, file_extension = os.path.splitext(file_path)
    if file_extension in [".jpg", ".png", ".bmp", ".jpeg"]:
        return pil_loader(file_path, convert_to_grayscale)
    elif file_extension in [".tif", ".tiff"]:
        return tiff_loader(file_path)
    else:
        raise ValueError("Invalid image. It should be `.jpg, .png, .bmp, .tif, .tiff, .jpeg`")


def image_invert(file_path, convert_to_grayscale=False):
    img = Image.open(file_path).convert("L")
    if convert_to_grayscale:
        img = ImageOps.invert(img)
    return np.asarray(img)


def stringify(obj):
    """stringify whatever object you have"""
    if isinstance(obj, list):
        response = ", ".join([stringify(o) for o in obj])
    elif isinstance(obj, dict):
        response = ", ".join([f"{k}:{stringify(v)}" for k, v in obj.items()])
    else:
        response = str(obj)

    return response


def parse_img_id(file_path, orients):
    """Parses direction, strip and coordinate components from a SpaceNet6 image filepath."""
    file_name = file_path.split("/")[-1]
    strip_name = "_".join(file_name.split("_")[-4:-2])
    direction = int(orients.loc[strip_name]["direction"])
    direction = torch.from_numpy(np.reshape(np.asarray([direction]), (1, 1, 1))).float()
    val = int(orients.loc[strip_name]["val"])
    strip = torch.Tensor(np.zeros((len(orients.index), 1, 1))).float()
    strip[val] = 1
    coord = np.asarray([orients.loc[strip_name]["coord_y"]])
    coord = torch.from_numpy(np.reshape(coord, (1, 1, 1))).float() - 0.5
    return direction, strip, coord


# Run this Function to split images into XxX pieces, and file out.txt containing the lists of patches
# Example call: split_images("./inria/images", "*.tif", "./inria/masks", "*.tif", "./inria/output", 500)
def split_images(images_dir, ext_images, masks_dir, ext_masks, output_dir, target_size):
    img_paths = glob.glob(os.path.join(images_dir, ext_images))
    mask_paths = glob.glob(os.path.join(masks_dir, ext_masks))
    file = open("list_patches.txt", "w")

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
                img_tile = img[y:y + target_size, x:x + target_size]
                mask_tile = mask[y:y + target_size, x:x + target_size]

                if img_tile.shape[0] == target_size and img_tile.shape[1] == target_size:
                    out_img_path = os.path.join(output_dir, "{}_{}.jpg".format(img_filename, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(output_dir, "{}_{}_m.png".format(mask_filename, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                    file.write("{}_{}".format(img_filename, k) + "\n")

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
    file.close()


def load_voc_format_dataset(dir_path, csv_file_path):
    """Loads a dataset in the Pascal VOC format. It expects a `multilabels.txt` file and `images` in the root folder"""

    # read labels
    multi_hot_labels = {}
    with open(csv_file_path, "rb") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.decode("utf-8")
            labels_list = line[line.find("\t") + 1:].split("\t")
            multi_hot_labels[line[: line.find("\t")]] = np.asarray(
                list((map(float, labels_list)))
            )

    images = []
    images_folder = os.path.expanduser(dir_path)
    # this ensures the image always have the same index numbers
    for root, _, fnames in sorted(os.walk(images_folder)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if fname[: fname.find(".")] in multi_hot_labels:
                multi_hot_label = multi_hot_labels[fname[: fname.find(".")]]
                item = (path, multi_hot_label)
                images.append(item)

    return images


def has_file_allowed_extension(file_path, extensions):
    """Checks if a file is an allowed extension.
    Args:
        file_path (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = file_path.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def load_folder_per_class_dataset(dir, extensions=None):
    if not extensions:
        raise ValueError("Please provide accepted extensions for image scanning.")

    images = []
    dir = os.path.expanduser(dir)
    classes = [
        item for item in os.listdir(dir) if os.path.isdir(os.path.join(dir, item))
    ]

    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(os.path.basename(os.path.normpath(root)), fname)
                    item = (path, target)
                    images.append(item)

    return images


def load_aitlas_format_dataset(file_path):
    """Reads the images from a CSV. Format: (image_path, class_name)"""
    data = []

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f)
        for index, row in enumerate(csv_reader):
            path = row[0]
            item = (path, row[1])
            data.append(item)

        return data


# Run this function to submit the masks to inria contest for semantic segmentation
# https://project.inria.fr/aerialimagelabeling/
def submit_inria_results(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith("_Buildings.png"):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(input_dir, file).replace("_Buildings.png", ".tif")
            command = "gdal_translate -of GTiff " + input_file + " " + output_file
            subprocess.call(command, shell=True)
            input_file = os.path.join(input_dir, file).replace("_Buildings.png", ".tif")
            output_file = os.path.join(output_dir, file).replace("_Buildings.png", ".tif")
            command = "gdal_translate --config GDAL_PAM_ENABLED NO -co COMPRESS=CCITTFAX4 -co NBITS=1 " \
                      + input_file + " " + output_file
            subprocess.call(command, shell=True)


def save_best_model(model, model_directory, epoch, optimizer, loss, start, run_id):
    """
    Saves the model on disk
    :param model_directory:
    :return:
    """
    if not os.path.isdir(os.path.join(model_directory, run_id)):
        os.makedirs(os.path.join(model_directory, run_id))

    timestamp = current_ts()
    checkpoint = os.path.join(
        model_directory, run_id, f"best_checkpoint_{timestamp}_{epoch}.pth.tar"
    )

    # create timestamped checkpoint
    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
            "start": start,
            "id": run_id,
        },
        checkpoint,
    )


def collate_fn(batch):
    return tuple(zip(*batch))

