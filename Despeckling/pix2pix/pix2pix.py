# -*- coding: utf-8 -*-
import scipy

# from tensorflow.keras.datasets import mnist
# from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.layers import Input, Reshape, Dropout, Concatenate, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

import tensorflow as tf



class Pix2Pix():
    def __init__(self,datadir):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Configure data loader
        self.datadir=datadir
        self.data_loader = DataLoader(datadir,img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_n = Input(shape=self.img_shape)
        img_o = Input(shape=self.img_shape)

        # Generate noise-free images from noised img
        fake_A = self.generator(img_n)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        validity = self.discriminator([fake_A, img_n])

        self.combined = Model(inputs=[img_n, img_o], outputs=[validity, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=2):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_n, imgs_o) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # generated noise-free img
                fake_A = self.generator.predict(imgs_n)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_o, imgs_n], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_n], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
 
                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                # Inputs: [imgs_n, imgs_o]
                # Outputs: [validity, fake_A]
                # Labels: [valid, imgs_o], where variable vaild is all 1 matrix
                g_loss = self.combined.train_on_batch([imgs_n, imgs_o], [valid, imgs_o])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time)) 

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
            if epoch % 5==0:
                self.generator.save('generator%d.h5' % (epoch))
                self.combined.save('combined%d.h5' % epoch)


    def sample_images(self, epoch, batch_i):
        os.makedirs('images/', exist_ok=True)
        r, c = 3, 3

        imgs_n,imgs_o = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_n)

        # gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5
        imgs_n =0.5 * imgs_n + 0.5
        imgs_o=0.5 * imgs_o + 0.5
        fake_A=0.5 * fake_A + 0.5
        gen_imgs = [imgs_n,fake_A,imgs_o]

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c, figsize=(20, 30))
        for i in range(r): #batch
            for j in range(c):
                axs[i,j].imshow(gen_imgs[j][i][:,:,0],cmap='gray')
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
        fig.savefig("images/%d_%d.png" % (epoch, batch_i))
        plt.close()
        


if __name__ == '__main__':
    # GPU 显存自动调用
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    datadir='C:\\travail\\dataset\\UCMerced_LandUse\\Images'
    gan = Pix2Pix(datadir)
    gan.train(epochs=200, batch_size=16, sample_interval=10)
