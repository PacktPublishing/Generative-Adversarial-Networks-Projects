import os
import time

import h5py
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imwrite
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Convolution2D, LeakyReLU, BatchNormalization, UpSampling2D, Dropout, Activation, Flatten, \
    Dense, Lambda, Reshape, concatenate
from keras.optimizers import Adam
import tensorflow as tf


def build_unet_generator():
    """
    Create the U-Net Generator using the hyperparameter values defined below
    """
    kernel_size = 4
    strides = 2
    leakyrelu_alpha = 0.2
    upsampling_size = 2
    dropout = 0.5
    output_channels = 1
    input_shape = (256, 256, 1)

    input_layer = Input(shape=input_shape)

    # Encoder Network

    # 1st Convolutional block in the encoder network
    encoder1 = Convolution2D(filters=64, kernel_size=kernel_size, padding='same',
                             strides=strides)(input_layer)
    encoder1 = LeakyReLU(alpha=leakyrelu_alpha)(encoder1)

    # 2nd Convolutional block in the encoder network
    encoder2 = Convolution2D(filters=128, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder1)
    encoder2 = BatchNormalization()(encoder2)
    encoder2 = LeakyReLU(alpha=leakyrelu_alpha)(encoder2)

    # 3rd Convolutional block in the encoder network
    encoder3 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder2)
    encoder3 = BatchNormalization()(encoder3)
    encoder3 = LeakyReLU(alpha=leakyrelu_alpha)(encoder3)

    # 4th Convolutional block in the encoder network
    encoder4 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder3)
    encoder4 = BatchNormalization()(encoder4)
    encoder4 = LeakyReLU(alpha=leakyrelu_alpha)(encoder4)

    # 5th Convolutional block in the encoder network
    encoder5 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder4)
    encoder5 = BatchNormalization()(encoder5)
    encoder5 = LeakyReLU(alpha=leakyrelu_alpha)(encoder5)

    # 6th Convolutional block in the encoder network
    encoder6 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder5)
    encoder6 = BatchNormalization()(encoder6)
    encoder6 = LeakyReLU(alpha=leakyrelu_alpha)(encoder6)

    # 7th Convolutional block in the encoder network
    encoder7 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder6)
    encoder7 = BatchNormalization()(encoder7)
    encoder7 = LeakyReLU(alpha=leakyrelu_alpha)(encoder7)

    # 8th Convolutional block in the encoder network
    encoder8 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same',
                             strides=strides)(encoder7)
    encoder8 = BatchNormalization()(encoder8)
    encoder8 = LeakyReLU(alpha=leakyrelu_alpha)(encoder8)

    # Decoder Network

    # 1st Upsampling Convolutional Block in the decoder network
    decoder1 = UpSampling2D(size=upsampling_size)(encoder8)
    decoder1 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same')(decoder1)
    decoder1 = BatchNormalization()(decoder1)
    decoder1 = Dropout(dropout)(decoder1)
    decoder1 = concatenate([decoder1, encoder7], axis=3)
    decoder1 = Activation('relu')(decoder1)

    # 2nd Upsampling Convolutional block in the decoder network
    decoder2 = UpSampling2D(size=upsampling_size)(decoder1)
    decoder2 = Convolution2D(filters=1024, kernel_size=kernel_size, padding='same')(decoder2)
    decoder2 = BatchNormalization()(decoder2)
    decoder2 = Dropout(dropout)(decoder2)
    decoder2 = concatenate([decoder2, encoder6])
    decoder2 = Activation('relu')(decoder2)

    # 3rd Upsampling Convolutional block in the decoder network
    decoder3 = UpSampling2D(size=upsampling_size)(decoder2)
    decoder3 = Convolution2D(filters=1024, kernel_size=kernel_size, padding='same')(decoder3)
    decoder3 = BatchNormalization()(decoder3)
    decoder3 = Dropout(dropout)(decoder3)
    decoder3 = concatenate([decoder3, encoder5])
    decoder3 = Activation('relu')(decoder3)

    # 4th Upsampling Convolutional block in the decoder network
    decoder4 = UpSampling2D(size=upsampling_size)(decoder3)
    decoder4 = Convolution2D(filters=1024, kernel_size=kernel_size, padding='same')(decoder4)
    decoder4 = BatchNormalization()(decoder4)
    decoder4 = concatenate([decoder4, encoder4])
    decoder4 = Activation('relu')(decoder4)

    # 5th Upsampling Convolutional block in the decoder network
    decoder5 = UpSampling2D(size=upsampling_size)(decoder4)
    decoder5 = Convolution2D(filters=1024, kernel_size=kernel_size, padding='same')(decoder5)
    decoder5 = BatchNormalization()(decoder5)
    decoder5 = concatenate([decoder5, encoder3])
    decoder5 = Activation('relu')(decoder5)

    # 6th Upsampling Convolutional block in the decoder network
    decoder6 = UpSampling2D(size=upsampling_size)(decoder5)
    decoder6 = Convolution2D(filters=512, kernel_size=kernel_size, padding='same')(decoder6)
    decoder6 = BatchNormalization()(decoder6)
    decoder6 = concatenate([decoder6, encoder2])
    decoder6 = Activation('relu')(decoder6)

    # 7th Upsampling Convolutional block in the decoder network
    decoder7 = UpSampling2D(size=upsampling_size)(decoder6)
    decoder7 = Convolution2D(filters=256, kernel_size=kernel_size, padding='same')(decoder7)
    decoder7 = BatchNormalization()(decoder7)
    decoder7 = concatenate([decoder7, encoder1])
    decoder7 = Activation('relu')(decoder7)

    # Last Convolutional layer
    decoder8 = UpSampling2D(size=upsampling_size)(decoder7)
    decoder8 = Convolution2D(filters=output_channels, kernel_size=kernel_size, padding='same')(decoder8)
    decoder8 = Activation('tanh')(decoder8)

    model = Model(inputs=[input_layer], outputs=[decoder8])
    return model


def build_patchgan_discriminator():
    """
    Create the PatchGAN discriminator using the hyperparameter values defined below
    """
    kernel_size = 4
    strides = 2
    leakyrelu_alpha = 0.2
    padding = 'same'
    num_filters_start = 64  # Number of filters to start with
    num_kernels = 100
    kernel_dim = 5
    patchgan_output_dim = (256, 256, 1)
    patchgan_patch_dim = (256, 256, 1)
    number_patches = int(
        (patchgan_output_dim[0] / patchgan_patch_dim[0]) * (patchgan_output_dim[1] / patchgan_patch_dim[1]))

    input_layer = Input(shape=patchgan_patch_dim)

    des = Convolution2D(filters=64, kernel_size=kernel_size, padding=padding, strides=strides)(input_layer)
    des = LeakyReLU(alpha=leakyrelu_alpha)(des)

    # Calculate the number of convolutional layers
    total_conv_layers = int(np.floor(np.log(patchgan_output_dim[1]) / np.log(2)))
    list_filters = [num_filters_start * min(total_conv_layers, (2 ** i)) for i in range(total_conv_layers)]

    # Next 7 Convolutional blocks
    for filters in list_filters[1:]:
        des = Convolution2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides)(des)
        des = BatchNormalization()(des)
        des = LeakyReLU(alpha=leakyrelu_alpha)(des)

    # Add a flatten layer
    flatten_layer = Flatten()(des)

    # Add the final dense layer
    dense_layer = Dense(units=2, activation='softmax')(flatten_layer)

    # Create the PatchGAN model
    model_patch_gan = Model(inputs=[input_layer], outputs=[dense_layer, flatten_layer])

    # Create a list of input layers equal to the number of patches
    list_input_layers = [Input(shape=patchgan_patch_dim) for _ in range(number_patches)]

    # Pass the patches through the PatchGAN network
    output1 = [model_patch_gan(patch)[0] for patch in list_input_layers]
    output2 = [model_patch_gan(patch)[1] for patch in list_input_layers]

    # In case of multiple patches, concatenate outputs to calculate perceptual loss
    if len(output1) > 1:
        output1 = concatenate(output1)
    else:
        output1 = output1[0]

    # In case of multiple patches, merge output2 as well
    if len(output2) > 1:
        output2 = concatenate(output2)
    else:
        output2 = output2[0]

    # Add a dense layer
    dense_layer2 = Dense(num_kernels * kernel_dim, use_bias=False, activation=None)

    # Add a lambda layer
    custom_loss_layer = Lambda(lambda x: K.sum(
        K.exp(-K.sum(K.abs(K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, pattern=(1, 2, 0)), 0)), 2)), 2))

    # Pass the output2 tensor through dense_layer2
    output2 = dense_layer2(output2)

    # Reshape the output2 tensor
    output2 = Reshape((num_kernels, kernel_dim))(output2)

    # Pass the output2 tensor through the custom_loss_layer
    output2 = custom_loss_layer(output2)

    # Finally concatenate output1 and output2
    output1 = concatenate([output1, output2])
    final_output = Dense(2, activation="softmax")(output1)

    # Create a discriminator model
    discriminator = Model(inputs=list_input_layers, outputs=[final_output])
    return discriminator


def build_adversarial_model(generator, discriminator):
    """
    Create an adversarial model
    """
    input_image_dim = (256, 256, 1)
    patch_dim = (256, 256)

    # Create an input layer
    input_layer = Input(shape=input_image_dim)

    # Use the generator network to generate images
    generated_images = generator(input_layer)

    # Extract patches from the generated images
    img_height, img_width = input_img_dim[:2]
    patch_height, patch_width = patch_dim

    row_idx_list = [(i * patch_height, (i + 1) * patch_height) for i in range(int(img_height / patch_height))]
    column_idx_list = [(i * patch_width, (i + 1) * patch_width) for i in range(int(img_width / patch_width))]

    generated_patches_list = []
    for row_idx in row_idx_list:
        for column_idx in column_idx_list:
            generated_patches_list.append(Lambda(lambda z: z[:, column_idx[0]:column_idx[1], row_idx[0]:row_idx[1], :],
                                                 output_shape=input_img_dim)(generated_images))

    discriminator.trainable = False

    # Pass the generated patches through the discriminator network
    dis_output = discriminator(generated_patches_list)

    # Create a model
    model = Model(inputs=[input_layer], outputs=[generated_images, dis_output])
    return model


"""
Data preprocessing methods
"""


def generate_and_extract_patches(images, facades, generator_model, batch_counter, patch_dim):
    # Alternatively, train the discriminator network on real and generated images
    if batch_counter % 2 == 0:
        # Generate fake images
        output_images = generator_model.predict(facades)

        # Create a batch of ground truth labels
        labels = np.zeros((output_images.shape[0], 2), dtype=np.uint8)
        labels[:, 0] = 1

    else:
        # Take real images
        output_images = images

        # Create a batch of ground truth labels
        labels = np.zeros((output_images.shape[0], 2), dtype=np.uint8)
        labels[:, 1] = 1

    patches = []
    for y in range(0, output_images.shape[0], patch_dim[0]):
        for x in range(0, output_images.shape[1], patch_dim[1]):
            image_patches = output_images[:, y: y + patch_dim[0], x: x + patch_dim[1], :]
            patches.append(np.asarray(image_patches, dtype=np.float32))

    return patches, labels


def save_images(real_images, real_sketches, generated_images, num_epoch, dataset_name, limit):
    real_sketches = real_sketches * 255.0
    real_images = real_images * 255.0
    generated_images = generated_images * 255.0

    # Save some images only
    real_sketches = real_sketches[:limit]
    generated_images = generated_images[:limit]
    real_images = real_images[:limit]

    # Create a stack of images
    X = np.hstack((real_sketches, generated_images, real_images))

    # Save stack of images
    imwrite('results/X_full_{}_{}.png'.format(dataset_name, num_epoch), X[0])


def load_dataset(data_dir, data_type, img_width, img_height):
    data_dir_path = os.path.join(data_dir, data_type)

    # Get all .h5 files containing training images
    facade_photos_h5 = [f for f in os.listdir(os.path.join(data_dir_path, 'images')) if '.h5' in f]
    facade_labels_h5 = [f for f in os.listdir(os.path.join(data_dir_path, 'facades')) if '.h5' in f]

    final_facade_photos = None
    final_facade_labels = None

    for index in range(len(facade_photos_h5)):
        facade_photos_path = data_dir_path + '/images/' + facade_photos_h5[index]
        facade_labels_path = data_dir_path + '/facades/' + facade_labels_h5[index]

        facade_photos = h5py.File(facade_photos_path, 'r')
        facade_labels = h5py.File(facade_labels_path, 'r')

        # Resize and normalize images
        num_photos = facade_photos['data'].shape[0]
        num_labels = facade_labels['data'].shape[0]

        all_facades_photos = np.array(facade_photos['data'], dtype=np.float32)
        all_facades_photos = all_facades_photos.reshape((num_photos, img_width, img_height, 1)) / 255.0

        all_facades_labels = np.array(facade_labels['data'], dtype=np.float32)
        all_facades_labels = all_facades_labels.reshape((num_labels, img_width, img_height, 1)) / 255.0

        if final_facade_photos is not None and final_facade_labels is not None:
            final_facade_photos = np.concatenate([final_facade_photos, all_facades_photos], axis=0)
            final_facade_labels = np.concatenate([final_facade_labels, all_facades_labels], axis=0)
        else:
            final_facade_photos = all_facades_photos
            final_facade_labels = all_facades_labels

    return final_facade_photos, final_facade_labels


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


def visualize_bw_image(img):
    """
    Visualize a black and white image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.axis("off")
    ax.set_title("Image")
    plt.show()


def save_bw_image(img, path):
    """
    Save a black and white image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.axis("off")
    ax.set_title("Image")
    plt.savefig(path)


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


if __name__ == '__main__':
    epochs = 500
    num_images_per_epoch = 400
    batch_size = 1
    img_width = 256
    img_height = 256
    num_channels = 1
    input_img_dim = (256, 256, 1)
    patch_dim = (256, 256)
    dataset_dir = "/Path/to/pix2pix/dataset/"

    common_optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    """
    Build and compile networks
    """
    # build and compile discriminator network
    patchgan_discriminator = build_patchgan_discriminator()
    patchgan_discriminator.compile(loss='binary_crossentropy', optimizer=common_optimizer)

    # build and compile the generator network
    unet_generator = build_unet_generator()
    unet_generator.compile(loss='mae', optimizer=common_optimizer)

    # Build and compile the adversarial model
    adversarial_model = build_adversarial_model(unet_generator, patchgan_discriminator)
    adversarial_model.compile(loss=['mae', 'binary_crossentropy'], loss_weights=[1E2, 1], optimizer=common_optimizer)

    """
    Load the training, testing and validation datasets
    """
    training_facade_photos, training_facade_labels = load_dataset(data_dir=dataset_dir, data_type='training',
                                                                  img_width=img_width, img_height=img_height)

    test_facade_photos, test_facade_labels = load_dataset(data_dir=dataset_dir, data_type='testing',
                                                          img_width=img_width, img_height=img_height)

    validation_facade_photos, validation_facade_labels = load_dataset(data_dir=dataset_dir, data_type='validation',
                                                                      img_width=img_width, img_height=img_height)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(unet_generator)
    tensorboard.set_model(patchgan_discriminator)

    print('Starting the training...')
    for epoch in range(0, epochs):
        print('Epoch {}'.format(epoch))

        dis_losses = []
        gen_losses = []

        batch_counter = 1
        start = time.time()

        num_batches = int(training_facade_photos.shape[0] / batch_size)

        # Train the networks for number of batches
        for index in range(int(training_facade_photos.shape[0] / batch_size)):
            print("Batch:{}".format(index))

            # Sample a batch of training and validation images
            train_facades_batch = training_facade_labels[index * batch_size:(index + 1) * batch_size]
            train_images_batch = training_facade_photos[index * batch_size:(index + 1) * batch_size]

            val_facades_batch = validation_facade_labels[index * batch_size:(index + 1) * batch_size]
            val_images_batch = validation_facade_photos[index * batch_size:(index + 1) * batch_size]

            patches, labels = generate_and_extract_patches(train_images_batch, train_facades_batch, unet_generator,
                                                           batch_counter, patch_dim)

            """
            Train the discriminator model
            """
            d_loss = patchgan_discriminator.train_on_batch(patches, labels)

            labels = np.zeros((train_images_batch.shape[0], 2), dtype=np.uint8)
            labels[:, 1] = 1

            """
            Train the adversarial model
            """
            g_loss = adversarial_model.train_on_batch(train_facades_batch, [train_images_batch, labels])

            # Increase the batch counter
            batch_counter += 1

            print("Discriminator loss:", d_loss)
            print("Generator loss:", g_loss)

            gen_losses.append(g_loss[1])
            dis_losses.append(d_loss)

        """
        Save losses to Tensorboard after each epoch
        """
        write_log(tensorboard, 'discriminator_loss', np.mean(dis_losses), epoch)
        write_log(tensorboard, 'generator_loss', np.mean(gen_losses), epoch)

        # After every 10th epoch, generate and save images for visualization
        if epoch % 10 == 0:
            # Sample a batch of validation datasets
            val_facades_batch = validation_facade_labels[0:5]
            val_images_batch = validation_facade_photos[0:5]

            # Generate images
            validation_generated_images = unet_generator.predict(val_facades_batch)

            # Save images
            save_images(val_images_batch, val_facades_batch, validation_generated_images, epoch, 'validation', limit=5)

    # Save models
    unet_generator.save_weights("generator.h5")
    patchgan_discriminator.save_weights("discriminator.h5")
