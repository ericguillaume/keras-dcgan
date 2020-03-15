import numpy as np
from PIL import Image
from keras.datasets import mnist
from tqdm import tqdm


def resize_grey_dataset(data, new_size, normalize=False):
    """
    :param data: shape: (m, IMG_SIZE, IMG_SIZE), should not have been normalized
    :param new_size:
    :param normalize:
    :return: shape: (m, IMG_SIZE, IMG_SIZE), same type as data
    """
    new_shape = (data.shape[0], new_size, new_size)
    result = np.zeros(new_shape)
    print("Resizing dataset to new_size: {}".format(new_size))
    for i, image_array in enumerate(tqdm(data)):
        image_array = np.squeeze(np.uint8(image_array))

        im = Image.fromarray(image_array, 'L')
        im = im.resize((new_size, new_size), Image.ANTIALIAS)
        image_array = np.asarray(im).astype(data.dtype)

        image_array = image_array[np.newaxis, :, :]
        result[i] = image_array
    print("result.shape = {}".format(result.shape))
    print("image_array.dtype = {}".format(image_array.dtype))
    if normalize:
        result = (result - 127.5) / 127.5
    return result


def load_mnist(normalize, resize=None, keep_only_2_digits=False):
    """
    :return: X_train.shape = (60000, 28, 28, 1)
            y_train.shape = (60000,)
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(np.float32)
    if resize:
        X_train = resize_grey_dataset(X_train, resize, normalize=False)
    if normalize:
        X_train = (X_train - 127.5) / 127.5
    if keep_only_2_digits:
        digits_kept_mask_train = np.logical_or(y_train == 0, y_train == 1)
        digits_kept_mask_test = np.logical_or(y_test == 0, y_test == 1)
        X_train = X_train[digits_kept_mask_train]
        y_train = y_train[digits_kept_mask_train]
        X_test = X_test[digits_kept_mask_test]
        y_test = y_test[digits_kept_mask_test]
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    return X_train, y_train
