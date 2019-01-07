import glob
import os

import numpy as np
import tensorflow as tf
from keras import Input
from keras.applications import VGG19
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, LeakyReLU, Add, Dense, PReLU, Flatten, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import preprocess_input
from keras_preprocessing.image import img_to_array, load_img
from scipy.misc import imsave, imread, imresize


def residual_block(x):
    """
    Residual block
    """
    filters = [64, 64]
    kernel_size = 3
    strides = 1
    padding = 'same'
    momentum = 0.8
    activation = "relu"

    res = Conv2D(filters=filters[0], kernel_size=kernel_size, strides=strides, padding=padding)(x)
    res = Activation(activation=activation)(res)
    res = BatchNormalization(momentum=momentum)(res)

    res = Conv2D(filters=filters[1], kernel_size=kernel_size, strides=strides, padding=padding)(res)
    res = BatchNormalization(momentum=momentum)(res)

    # Add res and x
    res = Add()([res, x])
    return res


def build_generator():
    """
    Create a generator network using the hyperparameter values defined below
    :return:
    """
    residual_blocks = 16
    momentum = 0.8
    channels = 3
    input_shape = (64, 64, channels)

    # Input Layer of the generator network
    input_layer = Input(shape=input_shape)

    # Add the pre-residual convolution layer
    gen1 = Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu')(input_layer)

    # Add 16 residual blocks
    res = residual_block(gen1)
    for i in range(residual_blocks - 1):
        res = residual_block(res)

    # Add the post-residual convolution layer
    gen2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
    gen2 = BatchNormalization(momentum=momentum)(gen2)

    # Take the sum of the output from the pre-residual and the post-residual convolution layers
    gen3 = Add()([gen2, gen1])

    # Add an upsampling block
    gen4 = UpSampling2D(size=2)(gen3)
    gen4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen4)
    gen4 = Activation('relu')(gen4)

    # Add another upsampling block
    gen5 = UpSampling2D(size=2)(gen4)
    gen5 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen5)
    gen5 = Activation('relu')(gen5)

    # Output convolution layer
    gen6 = Conv2D(filters=channels, kernel_size=9, strides=1,
                  padding='same')(gen5)
    output = Activation('tanh')(gen6)

    # Keras model
    model = Model(inputs=[input_layer], outputs=[output], name='generator')

    return model


def build_discriminator():
    """
    Create a discriminator network using the hyperparameter values defined below
    :return:
    """
    leakyrelu_alpha = 0.2
    momentum = 0.8
    input_shape = (256, 256, 3)

    input_layer = Input(shape=input_shape)

    # Add the first convolution block
    dis1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
    dis1 = LeakyReLU(alpha=leakyrelu_alpha)(dis1)

    # Add the 2nd convolution block
    dis2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
    dis2 = LeakyReLU(alpha=leakyrelu_alpha)(dis2)
    dis2 = BatchNormalization(momentum=momentum)(dis2)

    # Add the third convolution block
    dis3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
    dis3 = LeakyReLU(alpha=leakyrelu_alpha)(dis3)
    dis3 = BatchNormalization(momentum=momentum)(dis3)

    # Add the fourth convolution block
    dis4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
    dis4 = LeakyReLU(alpha=leakyrelu_alpha)(dis4)
    dis4 = BatchNormalization(momentum=0.8)(dis4)

    # Add the fifth convolution block
    dis5 = Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
    dis5 = LeakyReLU(alpha=leakyrelu_alpha)(dis5)
    dis5 = BatchNormalization(momentum=momentum)(dis5)

    # Add the sixth convolution block
    dis6 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
    dis6 = LeakyReLU(alpha=leakyrelu_alpha)(dis6)
    dis6 = BatchNormalization(momentum=momentum)(dis6)

    # Add the seventh convolution block
    dis7 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
    dis7 = LeakyReLU(alpha=leakyrelu_alpha)(dis7)
    dis7 = BatchNormalization(momentum=momentum)(dis7)

    # Add the eight convolution block
    dis8 = Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
    dis8 = LeakyReLU(alpha=leakyrelu_alpha)(dis8)
    dis8 = BatchNormalization(momentum=momentum)(dis8)

    # Add a dense layer
    dis9 = Dense(units=1024)(dis8)
    dis9 = LeakyReLU(alpha=0.2)(dis9)

    # Last dense layer - for classification
    output = Dense(units=1, activation='sigmoid')(dis9)

    model = Model(inputs=[input_layer], outputs=[output], name='discriminator')
    print(model.summary())
    return model


def build_vgg():
    """
    Build VGG network and extract features from an intermediate layer of VGG19 network
    """
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[9].output]
    img = Input(shape=(256, 256, 3))
    img_features = vgg(img)
    return Model(inputs=[img], outputs=[img_features])


def build_adversarial_model(generator, discriminator, vgg):
    """
    Build an adversarial model
    """
    input_low_resolution = Input(shape=(64, 64, 3))
    fake_hr_images = generator(input_low_resolution)

    fake_features = vgg(fake_hr_images)
    discriminator.trainable = False

    output = discriminator(fake_hr_images)
    model = Model(inputs=[input_low_resolution],
                  outputs=[output, fake_features])
    print(model.summary())
    return model


def load_images(images, target_size):
    """
    Load a batch of images
    """
    images_array = []
    for filepath in images:
        img = imread(filepath, mode='RGB').astype(np.float)
        img = imresize(img, target_size)
        images_array.append(img)

    return np.array(images_array)


def write_log(callback, name, value, batch_no):
    """
    Write logs to Tensorboard
    """
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


if __name__ == '__main__':
    # Define hyperparameters
    batch_size = 8
    low_resolution_shape = (64, 64, 3)
    high_resolution_shape = (256, 256, 3)
    epochs = 1000
    root_dir = "Path to the dataset directory"

    # Define optimizers
    dis_optimizer = Adam(0.0002, 0.5)
    gen_optimizer = Adam(0.0002, 0.5)
    vgg_optimizer = Adam(0.0002, 0.5)

    """
    Create and compile the networks
    """

    # VGG19
    vgg = build_vgg()
    vgg.trainable = False
    vgg.compile(loss='mse', optimizer=vgg_optimizer, metrics=['accuracy'])

    # Discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss=['mse'], optimizer=dis_optimizer)

    # Generator
    generator = build_generator()
    generator.compile(loss=['mse'], optimizer=gen_optimizer)

    # Adversarial model
    adversarial_model = build_adversarial_model(generator=generator, discriminator=discriminator, vgg=vgg)
    adversarial_model.compile(loss=['binary_crossentropy', 'mse'], loss_weights=[0.001, 1], optimizer=gen_optimizer)

    # Tensorboard
    tensorboard = TensorBoard(log_dir="./logs")
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    Load the dataset
    """

    train_hr_images = glob.glob(os.path.join(root_dir, "DIV2K_train_HR/*.*"))
    validation_hr_images = glob.glob(os.path.join(root_dir, "DIV2K_valid_HR/*.*"))

    train_lr_images = glob.glob(os.path.join(root_dir, "DIV2K_train_LR_bicubic/X4/*.*"))
    validation_lr_images = glob.glob(os.path.join(root_dir, "DIV2K_valid_LR_bicubic/X4/*.*"))

    """
    Train the model
    """

    for epoch in range(epochs):
        print("Epoch:", epoch)

        train_images = np.random.choice(train_hr_images, size=batch_size)
        validation_images = np.random.choice(validation_hr_images, size=batch_size)

        train_hr_batch = load_images(train_images, target_size=high_resolution_shape) / 127.5 - 1.0
        train_lr_batch = load_images(train_images, target_size=low_resolution_shape) / 127.5 - 1.0

        validation_hr_batch = load_images(validation_images, target_size=high_resolution_shape)
        validation_lr_batch = load_images(validation_images, target_size=low_resolution_shape)

        train_hr_batch = train_hr_batch.astype('float32')
        train_lr_batch = train_lr_batch.astype('float32')

        # Generate fake images
        fake_hr_images = generator.predict_on_batch(train_lr_batch)

        """
        Train the discriminator network
        """
        # With label smoothing
        y_real = np.ones((batch_size, 16, 16, 1), dtype='float32') * 0.9
        y_fake = np.zeros((batch_size, 16, 16, 1), dtype='float32') * 0.1

        d_real_loss = discriminator.train_on_batch([train_hr_batch], [y_real])
        d_fake_loss = discriminator.train_on_batch([fake_hr_images], [y_fake])

        d_loss = (d_real_loss + d_fake_loss) / 2
        print("d_real_loss:", d_real_loss)
        print("d_fake_loss:", d_fake_loss)
        print("d_loss:", d_loss)

        """
        Train the generator network
        """

        train_images = np.random.choice(train_hr_images, size=batch_size)
        train_hr_batch = load_images(train_images, target_size=high_resolution_shape) / 127.5 - 1.0
        train_hr_batch = train_hr_batch.astype('float32')

        # Extract features
        real_features = vgg.predict_on_batch(train_hr_batch)

        g_loss = adversarial_model.train_on_batch([train_lr_batch], [y_real, real_features])
        print("g_loss:", g_loss)

        # Save images after every epoch
        validation_lr_batch = validation_lr_batch / 127.5 - 1.0
        generated_images = generator.predict_on_batch(validation_lr_batch)

        for index1, img in enumerate(generated_images[:3]):
            imsave('results/image_fake_{}_{}.jpg'.format(epoch, index1), (img * 0.5) * 0.5)
            imsave('results/image_real_{}_{}.jpg'.format(epoch, index1), (validation_lr_batch[index1] * 0.5) * 0.5)

        # Add average loss to Tensorboard
        write_log(tensorboard, 'g_loss', g_loss[1], epoch)
        write_log(tensorboard, 'd_loss', d_loss, epoch)

    # Save models
    generator.save_weights("generator.h5")
    discriminator.save_weights("discriminator.h5")
