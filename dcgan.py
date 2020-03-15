import argparse

import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import SGD

from model import get_default_discriminator, get_default_generator
from tools import combine_images, Tensorboard


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


EPOCHS = 1  # 100


def train(batch_size):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])

    d = get_default_discriminator()
    g = get_default_generator()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    board = Tensorboard("default_logs")
    for epoch in range(EPOCHS):
        print("Epoch is", epoch)
        batches = int(X_train.shape[0] / batch_size)
        print("Number of batches", batches)

        for index in range(int(X_train.shape[0] / batch_size)):
            step = (epoch * batches) + index
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            image_batch = X_train[index * batch_size:(index + 1) * batch_size]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                generated_images_output = "generated/" + str(epoch) + "_" + str(index) + ".png"
                Image.fromarray(image.astype(np.uint8)).save(generated_images_output)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * batch_size + [0] * batch_size
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (batch_size, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))

            board.log_scalar("discriminator_loss", d_loss, step)
            board.log_scalar("generator_loss", g_loss, step)

            if index % 10 == 9:
                g.save_weights('generated/model/generator', True)
                d.save_weights('generated/model/discriminator', True)


def generate(batch_size, nice=False):
    g = get_default_generator()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = get_default_discriminator()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (batch_size * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
