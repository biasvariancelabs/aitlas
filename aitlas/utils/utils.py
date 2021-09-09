import importlib
import os
from time import time
import glob
import cv2
import numpy as np
import tifffile
import torch

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
    if file_extension in [".jpg", ".png", ".bmp"]:
        return pil_loader(file_path, convert_to_grayscale)
    elif file_extension in [".tif", ".tiff"]:
        return tiff_loader(file_path)
    else:
        raise ValueError("Invalid image. It should be `.jpg, .png, .bmp, .tif, .tiff`")


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
# Example call: split_images("./inria/images", "./inria/masks", "./inria/output", 500)
def split_images(images_dir, masks_dir, output_dir, target_size):
    img_paths = glob.glob(os.path.join(images_dir, "*.tif"))
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))
    file = open("list_patches.txt", "w")

    img_paths.sort()
    mask_paths.sort()

    os.makedirs(output_dir)
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        print(img_path, mask_path)

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

                    file.write(img_filename + "\n")

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
        file.close()

