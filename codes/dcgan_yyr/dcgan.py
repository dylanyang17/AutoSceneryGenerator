import numpy as np
import matplotlib.pyplot as plt
import keras

class DCGAN():
    def __init__(self):
        self.lr = 0.001  # learning rate
        self.bn_momentum = 0.8
        self.channels = 3
        self.img_shape = (128, 128, self.channels)
        self.noise_shape = (100,)
        self.loss_func = 'binary_crossentropy'
        self.data_path = '../data128/pics.npy'

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.loss_func, optimizer=keras.optimizers.Adam(lr=self.lr),
                                   metrics=['accuracy'])
        self.generator = self.build_generator()

        z = keras.layers.Input(shape=self.noise_shape)
        img = self.generator(z)
        self.discriminator.trainable = False
        score = self.discriminator(img)
        self.combined = keras.models.Model(z, score)
        self.combined.compile(loss=self.loss_func, optimizer=keras.optimizers.Adam(lr=self.lr))

    def build_discriminator(self):
        # 构建 discriminator
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(input_shape=self.img_shape, filters=64, kernel_size=4, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        # model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        # model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        # model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=4, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        # model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='tanh'))

        img = keras.layers.Input(shape=self.img_shape)
        score = model(img)
        return keras.models.Model(img, score)

    def build_generator(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(512*8*8, input_shape=self.noise_shape))
        model.add(keras.layers.Reshape((8, 8, 512)))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(filters=256, kernel_size=4, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(filters=128, kernel_size=4, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(filters=64, kernel_size=4, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2D(filters=self.channels, kernel_size=4, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('tanh'))

        z = keras.layers.Input(shape=self.noise_shape)
        img = model(z)
        return keras.models.Model(z, img)

    def train(self, epochs, batch_size, save_interval):
        np.load(self.data_path)

if __name__ == '__main__':
    dcgan = DCGAN()

    # dcgan.train()
