import glob
import io
import math
import time

import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential, Input, Model
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from scipy.misc import imread, imsave
from scipy.stats import entropy

K.set_image_dim_ordering('tf')

np.random.seed(1337)


def build_generator():
    gen_model = Sequential()

    gen_model.add(Dense(input_dim=100, output_dim=2048))
    gen_model.add(ReLU())

    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(Activation('tanh'))
    return gen_model


def build_discriminator():
    dis_model = Sequential()
    dis_model.add(
        Conv2D(128, (5, 5),
               padding='same',
               input_shape=(64, 64, 3))
    )
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(LeakyReLU(alpha=0.2))

    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))

    return dis_model


def build_adversarial_model(gen_model, dis_model):
    model = Sequential()
    model.add(gen_model)
    dis_model.trainable = False
    model.add(dis_model)
    return model


def write_log(callback, name, loss, batch_no):
    """
    Write training summary to TensorBoard
    """
    # for name, value in zip(names, logs):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = loss
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def calculate_inception_score(images_path, batch_size=1, splits=10):
    # Create an instance of InceptionV3
    model = InceptionResNetV2()

    images = None
    for image_ in glob.glob(images_path):
        # Load image
        loaded_image = image.load_img(image_, target_size=(299, 299))

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Another another dimension (Add batch dimension)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Concatenate all images into one tensor
        if images is None:
            images = loaded_image
        else:
            images = np.concatenate([images, loaded_image], axis=0)

    # Calculate number of batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size

    probs = None

    # Use InceptionV3 to calculate probabilities
    for i in range(num_batches):
        image_batch = images[i * batch_size:(i + 1) * batch_size, :, :, :]
        prob = model.predict(preprocess_input(image_batch))

        if probs is None:
            probs = prob
        else:
            probs = np.concatenate([prob, probs], axis=0)

    # Calculate Inception scores
    divs = []
    split_size = probs.shape[0] // splits

    for i in range(splits):
        prob_batch = probs[(i * split_size):((i + 1) * split_size), :]
        p_y = np.expand_dims(np.mean(prob_batch, 0), 0)
        div = prob_batch * (np.log(prob_batch / p_y))
        div = np.mean(np.sum(div, 1))
        divs.append(np.exp(div))

    return np.mean(divs), np.std(divs)


def calculate_mode_score(gen_images_path, real_images_path, batch_size=32, splits=10):
    # Create an instance of InceptionV3
    model = InceptionResNetV2()

    # Load real images
    real_images = None
    for image_ in glob.glob(real_images_path):
        # Load image
        loaded_image = image.load_img(image_, target_size=(299, 299))

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Another another dimension (Add batch dimension)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Concatenate all images into one tensor
        if real_images is None:
            real_images = loaded_image
        else:
            real_images = np.concatenate([real_images, loaded_image], axis=0)

    # Load generated images
    gen_images = None
    for image_ in glob.glob(gen_images_path):
        # Load image
        loaded_image = image.load_img(image_, target_size=(299, 299))

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Another another dimension (Add batch dimension)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Concatenate all images into one tensor
        if gen_images is None:
            gen_images = loaded_image
        else:
            gen_images = np.concatenate([gen_images, loaded_image], axis=0)

    # Calculate number of batches for generated images
    gen_num_batches = (gen_images.shape[0] + batch_size - 1) // batch_size
    gen_images_probs = None
    # Use InceptionV3 to calculate probabilities of generated images
    for i in range(gen_num_batches):
        image_batch = gen_images[i * batch_size:(i + 1) * batch_size, :, :, :]
        prob = model.predict(preprocess_input(image_batch))

        if gen_images_probs is None:
            gen_images_probs = prob
        else:
            gen_images_probs = np.concatenate([prob, gen_images_probs], axis=0)

    # Calculate number of batches for real images
    real_num_batches = (real_images.shape[0] + batch_size - 1) // batch_size
    real_images_probs = None
    # Use InceptionV3 to calculate probabilities of real images
    for i in range(real_num_batches):
        image_batch = real_images[i * batch_size:(i + 1) * batch_size, :, :, :]
        prob = model.predict(preprocess_input(image_batch))

        if real_images_probs is None:
            real_images_probs = prob
        else:
            real_images_probs = np.concatenate([prob, real_images_probs], axis=0)

    # KL-Divergence: compute kl-divergence and mean of it
    num_gen_images = len(gen_images)
    split_scores = []

    for j in range(splits):
        gen_part = gen_images_probs[j * (num_gen_images // splits): (j + 1) * (num_gen_images // splits), :]
        real_part = real_images_probs[j * (num_gen_images // splits): (j + 1) * (num_gen_images // splits), :]
        gen_py = np.mean(gen_part, axis=0)
        real_py = np.mean(real_part, axis=0)
        scores = []
        for i in range(gen_part.shape[0]):
            scores.append(entropy(gen_part[i, :], gen_py))

        split_scores.append(np.exp(np.mean(scores) - entropy(gen_py, real_py)))

    final_mean = np.mean(split_scores)
    final_std = np.std(split_scores)

    return final_mean, final_std


def denormalize(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def normalize(img):
    return (img - 127.5) / 127.5


def visualize_rgb(img):
    """
    Visualize a rgb image
    :param img: RGB image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")
    plt.show()


def save_rgb_img(img, path):
    """
    Save a lab image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("RGB Image")

    plt.savefig(path)
    plt.close()


def train():
    start_time = time.time()
    dataset_dir = "/path/to/dataset/directory/*.*"
    batch_size = 128
    z_shape = 100
    epochs = 10000
    dis_learning_rate = 0.005
    gen_learning_rate = 0.005
    dis_momentum = 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True

    dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)
    gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)

    # Load images
    all_images = []
    for index, filename in enumerate(glob.glob(dataset_dir)):
        all_images.append(imread(filename, flatten=False, mode='RGB'))

    X = np.array(all_images)
    X = (X - 127.5) / 127.5
    X = X.astype(np.float32)

    dis_model = build_discriminator()
    dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    gen_model = build_generator()
    gen_model.compile(loss='mse', optimizer=gen_optimizer)

    adversarial_model = build_adversarial_model(gen_model, dis_model)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), write_images=True, write_grads=True, write_graph=True)
    tensorboard.set_model(gen_model)
    tensorboard.set_model(dis_model)

    for epoch in range(epochs):
        print("--------------------------")
        print("Epoch:{}".format(epoch))

        dis_losses = []
        gen_losses = []

        num_batches = int(X.shape[0] / batch_size)

        print("Number of batches:{}".format(num_batches))
        for index in range(num_batches):
            print("Batch:{}".format(index))

            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            # z_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            generated_images = gen_model.predict_on_batch(z_noise)

            # visualize_rgb(generated_images[0])

            """
            Train the discriminator model
            """

            dis_model.trainable = True

            image_batch = X[index * batch_size:(index + 1) * batch_size]

            y_real = np.ones((batch_size, )) * 0.9
            y_fake = np.zeros((batch_size, )) * 0.1

            dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

            d_loss = (dis_loss_real+dis_loss_fake)/2
            print("d_loss:", d_loss)

            dis_model.trainable = False

            """
            Train the generator model(adversarial model)
            """
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            # z_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            g_loss = adversarial_model.train_on_batch(z_noise, y_real)
            print("g_loss:", g_loss)

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        """
        Sample some images and save them
        """
        if epoch % 100 == 0:
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            gen_images1 = gen_model.predict_on_batch(z_noise)

            for img in gen_images1[:2]:
                save_rgb_img(img, "results/one_{}.png".format(epoch))

        print("Epoch:{}, dis_loss:{}".format(epoch, np.mean(dis_losses)))
        print("Epoch:{}, gen_loss: {}".format(epoch, np.mean(gen_losses)))

        """
        Save losses to Tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

    """
    Save models
    """
    gen_model.save("generator_model.h5")
    dis_model.save("generator_model.h5")

    print("Time:", (time.time() - start_time))


if __name__ == '__main__':
    train()
