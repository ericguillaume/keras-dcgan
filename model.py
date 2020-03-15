from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


def get_default_generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def get_default_discriminator():
    model = Sequential()
    model.add(
        Conv2D(64, (5, 5),
               padding='same',
               input_shape=(28, 28, 1))
    )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('tanh'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def get_eric_generator():
    model = Sequential()

    model.add(Dense(1024, input_shape=(100,)))
    model.add(Activation("tanh"))

    model.add(Dense(7 * 7 * 128))
    model.add(Activation("tanh"))
    model.add(Reshape(target_shape=(7, 7, 128)))

    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(32, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(UpSampling2D(size=(2, 2)))

    # (5, 5) car on a besoin du CONTEXTE LOCAL pour preciser l'image localement
    # est ce le cas plus au bas niveau ?
    model.add(Conv2D(1, (5, 5), padding="same"))
    model.add(Activation("tanh"))

    return model


def get_eric_discriminator():
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28, 28, 1)))
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(Activation("tanh"))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Reshape(target_shape=(7 * 7 * 128,)))

    model.add(Dense(1024))
    model.add(Activation("tanh"))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))  # verifier et pk ca ???   ou pas softmax ??

    return model