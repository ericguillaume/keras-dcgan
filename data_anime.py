import glob
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_all_images_in_folder(folder_path, max_size, extensions):
    """
    files should have images of same size, constant for now
    """

    filenames = []
    for ext in extensions:
        filenames += glob.glob("{}/**/*.{}".format(folder_path, ext))

    if not len(filenames):
        raise ValueError("{} contains no images with extesions {}".format(folder_path, extensions))

    images = np.zeros((max_size, 96, 96, 3))
    count = 0
    print("loading images from {}".format(folder_path))
    for filename in tqdm(filenames):
        if count >= max_size:
            break

        im = Image.open(filename)
        image_array = np.asarray(im)
        if not image_array.shape == (96, 96, 3):
            continue

        images[count] = image_array

        count += 1
    images = images[0:count]
    return images


def find_anime_faces_5k_filepath():
    filepath = "/home/eric/dev/data/anime-faces-processed_5k"
    if os.path.exists(filepath):
        return filepath
    return "/Users/eric/dev/data/anime-faces-processed_5k"


def load_anime_faces_processed(max_size):
    # warning c'est /Users locally
    return load_all_images_in_folder(find_anime_faces_5k_filepath(), max_size, ["jpg", "png"])
