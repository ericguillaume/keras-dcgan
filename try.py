from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
exit()




def generator_model():
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

def discriminator_model():
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

model = generator_model()
model.summary()
model = discriminator_model()
model.summary()
