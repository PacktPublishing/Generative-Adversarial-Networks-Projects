import glob
import os
import time

import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt


def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    """
    z_size = 200
    gen_filters = [512, 256, 128, 64, 1]
    gen_kernel_sizes = [4, 4, 4, 4, 4]
    gen_strides = [1, 2, 2, 2, 2]
    gen_input_shape = (1, 1, 1, z_size)
    gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
    gen_convolutional_blocks = 5

    input_layer = Input(shape=gen_input_shape)

    # First 3D transpose convolution(or 3D deconvolution) block
    a = Deconv3D(filters=gen_filters[0],
                 kernel_size=gen_kernel_sizes[0],
                 strides=gen_strides[0])(input_layer)
    a = BatchNormalization()(a, training=True)
    a = Activation(activation='relu')(a)

    # Next 4 3D transpose convolution(or 3D deconvolution) blocks
    for i in range(gen_convolutional_blocks - 1):
        a = Deconv3D(filters=gen_filters[i + 1],
                     kernel_size=gen_kernel_sizes[i + 1],
                     strides=gen_strides[i + 1], padding='same')(a)
        a = BatchNormalization()(a, training=True)
        a = Activation(activation=gen_activations[i + 1])(a)

    gen_model = Model(inputs=[input_layer], outputs=[a])
    return gen_model


def build_discriminator():
    """
    Create a Discriminator Model using hyperparameters values defined as follows
    """

    dis_input_shape = (64, 64, 64, 1)
    dis_filters = [64, 128, 256, 512, 1]
    dis_kernel_sizes = [4, 4, 4, 4, 4]
    dis_strides = [2, 2, 2, 2, 1]
    dis_paddings = ['same', 'same', 'same', 'same', 'valid']
    dis_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    dis_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu',
                       'leaky_relu', 'sigmoid']
    dis_convolutional_blocks = 5

    dis_input_layer = Input(shape=dis_input_shape)

    # The first 3D Convolutional block
    a = Conv3D(filters=dis_filters[0],
               kernel_size=dis_kernel_sizes[0],
               strides=dis_strides[0],
               padding=dis_paddings[0])(dis_input_layer)
    # a = BatchNormalization()(a, training=True)
    a = LeakyReLU(dis_alphas[0])(a)

    # Next 4 3D Convolutional Blocks
    for i in range(dis_convolutional_blocks - 1):
        a = Conv3D(filters=dis_filters[i + 1],
                   kernel_size=dis_kernel_sizes[i + 1],
                   strides=dis_strides[i + 1],
                   padding=dis_paddings[i + 1])(a)
        a = BatchNormalization()(a, training=True)
        if dis_activations[i + 1] == 'leaky_relu':
            a = LeakyReLU(dis_alphas[i + 1])(a)
        elif dis_activations[i + 1] == 'sigmoid':
            a = Activation(activation='sigmoid')(a)

    dis_model = Model(inputs=[dis_input_layer], outputs=[a])
    return dis_model


def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


"""
Load datasets
"""


def get3DImages(data_dir):
    all_files = np.random.choice(glob.glob(data_dir), size=10)
    # all_files = glob.glob(data_dir)
    all_volumes = np.asarray([getVoxelsFromMat(f) for f in all_files], dtype=np.bool)
    return all_volumes


def getVoxelsFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def saveFromVoxels(voxels, path):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.savefig(path)


def plotAndSaveVoxel(file_path, voxel):
    """
    Plot a voxel
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxel, edgecolor="red")
    # plt.show()
    plt.savefig(file_path)


if __name__ == '__main__':
    """
    Specify Hyperparameters
    """
    object_name = "chair"
    data_dir = "/Path/to/3DShapeNets/volumetric_data/" \
               "{}/30/train/*.mat".format(object_name)
    gen_learning_rate = 0.0025
    dis_learning_rate = 10e-5
    beta = 0.5
    batch_size = 1
    z_size = 200
    epochs = 10
    MODE = "train"

    """
    Create models
    """
    gen_optimizer = Adam(lr=gen_learning_rate, beta_1=beta)
    dis_optimizer = Adam(lr=dis_learning_rate, beta_1=beta)

    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    discriminator.trainable = False

    input_layer = Input(shape=(1, 1, 1, z_size))
    generated_volumes = generator(input_layer)
    validity = discriminator(generated_volumes)
    adversarial_model = Model(inputs=[input_layer], outputs=[validity])
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    print("Loading data...")
    volumes = get3DImages(data_dir=data_dir)
    volumes = volumes[..., np.newaxis].astype(np.float)
    print("Data loaded...")

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    labels_real = np.reshape(np.ones((batch_size,)), (-1, 1, 1, 1, 1))
    labels_fake = np.reshape(np.zeros((batch_size,)), (-1, 1, 1, 1, 1))

    if MODE == 'train':
        for epoch in range(epochs):
            print("Epoch:", epoch)

            gen_losses = []
            dis_losses = []

            number_of_batches = int(volumes.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                volumes_batch = volumes[index * batch_size:(index + 1) * batch_size, :, :, :]

                # Next, generate volumes using the generate network
                gen_volumes = generator.predict_on_batch(z_sample)

                """
                Train the discriminator network
                """
                discriminator.trainable = True
                if index % 2 == 0:
                    loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
                    loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

                    d_loss = 0.5 * np.add(loss_real, loss_fake)
                    print("d_loss:{}".format(d_loss))

                else:
                    d_loss = 0.0

                discriminator.trainable = False
                """
                Train the generator network
                """
                z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                g_loss = adversarial_model.train_on_batch(z, labels_real)
                print("g_loss:{}".format(g_loss))

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

                # Every 10th mini-batch, generate volumes and save them
                if index % 10 == 0:
                    z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
                    generated_volumes = generator.predict(z_sample2, verbose=3)
                    for i, generated_volume in enumerate(generated_volumes[:5]):
                        voxels = np.squeeze(generated_volume)
                        voxels[voxels < 0.5] = 0.
                        voxels[voxels >= 0.5] = 1.
                        saveFromVoxels(voxels, "results/img_{}_{}_{}".format(epoch, index, i))

            # Write losses to Tensorboard
            write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
            write_log(tensorboard, 'd_loss', np.mean(dis_losses), epoch)

        """
        Save models
        """
        generator.save_weights(os.path.join("models", "generator_weights.h5"))
        discriminator.save_weights(os.path.join("models", "discriminator_weights.h5"))

    if MODE == 'predict':
        # Create models
        generator = build_generator()
        discriminator = build_discriminator()

        # Load model weights
        generator.load_weights(os.path.join("models", "generator_weights.h5"), True)
        discriminator.load_weights(os.path.join("models", "discriminator_weights.h5"), True)

        # Generate 3D models
        z_sample = np.random.normal(0, 1, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
        generated_volumes = generator.predict(z_sample, verbose=3)

        for i, generated_volume in enumerate(generated_volumes[:2]):
            voxels = np.squeeze(generated_volume)
            voxels[voxels < 0.5] = 0.
            voxels[voxels >= 0.5] = 1.
            saveFromVoxels(voxels, "results/gen_{}".format(i))
