import numpy as np
from PIL import Image
from keras import optimizers
from keras.models import Sequential
from tqdm import tqdm

from data import load_mnist
from data_anime import load_anime_faces_processed
from dcgan import combine_images
from model import get_default_generator, get_default_discriminator
from tools import Tensorboard


'''
    * deployment script works with branch name
    * convergence
        * gradient avg values
        * D/G accuracy
    
    * anime
    
    * IDEA: garder historique des generated pour entrainer discriminator ???
'''


def get_model():
    # generator
    generator = get_default_generator()
    generator.summary()
    sgd_g = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # peut on avoir meme sgd partout ????
    generator.compile(optimizer=sgd_g, loss='binary_crossentropy')

    # discriminator
    discriminator = get_default_discriminator()
    discriminator.summary()
    sgd_d = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    discriminator.trainable = True
    discriminator.compile(optimizer=sgd_d, loss='binary_crossentropy')

    # generator_with_d
    generator_with_d = Sequential()
    generator_with_d.add(generator)
    generator_with_d.add(discriminator)

    # binary_crossentropy ???     0.67 que vaut ? et 0.8 sans cas generated
    discriminator.trainable = False
    sgd_gwd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    generator_with_d.compile(optimizer=sgd_gwd, loss='binary_crossentropy')

    return generator, discriminator, generator_with_d


def log_images(generated_images, field1, field2):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    generated_images_output = "generated/" + str(field1) + "_" + str(field2) + ".png"
    Image.fromarray(image.astype(np.uint8)).save(generated_images_output)


EPOCHS = 100
BATCH_SIZE = 64
IMAGE_SIZE = 96
IMAGE_COLORS_DIMENSIONS = 3

GENERATOR_INPUT_DIM = 100

# X_train, y_train = load_mnist(normalize=True, resize=RESIZE)
X_train = load_anime_faces_processed(12000)
print("X_train.shape = {}".format(X_train.shape))
generator, discriminator, generator_with_d = get_model()

logger = Tensorboard("logs")
for epoch in range(EPOCHS):
    print("epoch = {}".format(epoch))
    batches = int(X_train.shape[0] / BATCH_SIZE)
    for i in tqdm(range(batches)):
        step = (epoch * batches) + i
        print("step = {}".format(step))

        noise = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, GENERATOR_INPUT_DIM))
        generated = generator.predict(noise)
        print("generated.shape = {}".format(generated.shape))
        if i % 200 == 0:
            log_images(generated, epoch, i)
            log_images(generated, "last", "last")

        start_idx = i * BATCH_SIZE
        end_idx = (i + 1) * BATCH_SIZE
        X = X_train[start_idx:end_idx] + np.random.normal(loc=0.0, scale=1e-1, size=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_COLORS_DIMENSIONS))
        X = np.concatenate((X, generated))
        y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)

        discriminator.trainable = True
        discriminator_loss = discriminator.train_on_batch(X, y)  # reset_metrics=True ??  => metrics only for the batch
        logger.log_scalar("discriminator_loss", discriminator_loss, step)
        print("discriminator_loss = {}".format(discriminator_loss))

        y = [1] * BATCH_SIZE
        discriminator.trainable = False
        generator_loss = generator_with_d.train_on_batch(noise, y)
        logger.log_scalar("generator_loss", generator_loss, step)
        print("generator_loss = {}".format(generator_loss))

