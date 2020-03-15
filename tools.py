import math

import numpy as np
import tensorflow as tf


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image


class Tensorboard:

    def __init__(self, dirname):
        log_dir = "/tmp/{}".format(dirname)
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, name, value, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)
