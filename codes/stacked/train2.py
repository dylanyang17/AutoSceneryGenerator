from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from dataset import getDatasets32, getDatasets64, getDatasets128
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Add, Concatenate

from matplotlib import pyplot
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resBlock(inputShape, channel):
    x = Input(shape=inputShape)
    residual = x
    out = Conv2D(channel, (3, 3), strides=(1, 1), padding='same')(x)
    out = LeakyReLU(alpha=0.2)(out)
    out = Conv2D(channel, (3, 3), strides=(1, 1), padding='same')(out)
    out = Add()([out, residual])
    model = Model(inputs=x, outputs=out)
    return model


def define_discriminator_1(in_shape=(32, 32, 3)):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model


def define_discriminator_2(in_shape=(64, 64, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.00005, beta_1=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model


def define_generator_1(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


def define_generator_2(latent_dim):

    stage_I_result = Input((32, 32, 3))
    res = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(stage_I_result)
    out1 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(stage_I_result)
    out1 = LeakyReLU(alpha=0.2)(out1)
    out1 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(out1)
    out1 = LeakyReLU(alpha=0.2)(out1)  # [8, 8, 256]

    n_nodes = 64 * 8 * 8
    noise = Input((latent_dim,))
    out2 = Dense(n_nodes, input_dim=latent_dim)(noise)
    out2 = LeakyReLU(alpha=0.2)(out2)
    out2 = Reshape((8, 8, 64))(out2)
    merged = Concatenate()([out1, out2])
    out = resBlock(inputShape=(8, 8, 256+64), channel=256+64)(merged)

    # upsample to 16x16
    out = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(out)
    out = LeakyReLU(alpha=0.2)(out)

    out = resBlock(inputShape=(16, 16, 128), channel=128)(out)

    # upsample to 32x32
    out = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(out)
    out = LeakyReLU(alpha=0.2)(out)
    out = Add()([res, out])
    out = resBlock(inputShape=(32, 32, 128), channel=128)(out)

    # upsample to 64x64
    out = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(out)
    out = LeakyReLU(alpha=0.2)(out)

    out = resBlock(inputShape=(64, 64, 128), channel=128)(out)

    # output layer
    out = Conv2D(3, (3, 3), activation='tanh', padding='same')(out)

    model = Model(inputs=[stage_I_result, noise], outputs=out)
    return model


def define_gan_1(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0003, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def define_gan_2(g_model, d_model):
    d_model.trainable = False
    x = [Input((32, 32, 3)), Input((100,))]
    out = g_model(x)
    out = d_model(out)
    model = Model(inputs=x, outputs=out)

    # compile model
    opt = Adam(lr=0.00001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def load_real_samples():
    trainX32 = getDatasets32()
    trainX64 = getDatasets64()
    trainX128 = getDatasets128()

    X32 = trainX32.astype('float32')
    # scale from [0,255] to [-1,1]
    X32 = (X32 - 127.5) / 127.5

    X64 = trainX64.astype('float32')
    # scale from [0,255] to [-1,1]
    X64 = (X64 - 127.5) / 127.5

    X128 = trainX128.astype('float32')
    # scale from [0,255] to [-1,1]
    X128 = (X128 - 127.5) / 127.5
    return X32, X64, X128


def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input



def generate_fake_samples_1(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


def generate_fake_samples_2(g_model, stage_I_result, latent_dim, n_samples):
    # generate points in latent space
    noise = generate_latent_points(latent_dim, n_samples)
    x_input = [stage_I_result, noise]
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

# create and save a plot of generated images


def save_plot(examples, epoch, n=8):
    # scale from [-1,1] to [0,1]
    examples1, examples2 = examples
    examples1 = (examples1 + 1) / 2.0
    examples2 = (examples2 + 1) / 2.0
    # plot images
    for i in range(n):
        # define subplot
        pyplot.subplot(2, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples1[i])
        pyplot.subplot(2, n, n + 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples2[i])
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

# evaluate the discriminator, plot generated images, save generator model


def summarize_performance(epoch, g_model_1, d_model_1, g_model_2, d_model_2, dataset1, dataset2, n_samples=150):
    # GAN1
    X_real, y_real = generate_real_samples(dataset1, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model_1.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples_1(g_model_1, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model_1.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>GAN[1] Accuracy real: %.0f%%, fake: %.0f%%' %
          (acc_real*100, acc_fake*100))
    # save plot

    X_real2, y_real2 = generate_real_samples(dataset2, n_samples)
    _, acc_real2 = d_model_2.evaluate(X_real2, y_real2, verbose=0)
    x_fake2, y_fake2 = generate_fake_samples_2(
        g_model_2, x_fake, latent_dim, n_samples)
    _, acc_fake2 = d_model_2.evaluate(x_fake2, y_fake2, verbose=0)

    print('>GAN[2] Accuracy real: %.0f%%, fake: %.0f%%' %
          (acc_real2*100, acc_fake2*100))

    save_plot([x_fake, x_fake2], epoch)
    # save the generator model tile file
    g_filename1 = 'gen1_model{}.h5'.format(epoch+1)
    d_filename1 = 'dis1_model{}.h5'.format(epoch+1)
    g_filename2 = 'gen2_model{}.h5'.format(epoch+1)
    d_filename2 = 'dis2_model{}.h5'.format(epoch+1)
    g_model_1.save(g_filename1)
    d_model_1.save(d_filename1)
    g_model_2.save(g_filename2)
    d_model_2.save(d_filename2)

# train the generator and discriminator


def train(g_model_1, d_model_1, g_model_2, d_model_2, gan_model_1, gan_model_2, dataset1, dataset2, latent_dim, n_epochs=100, n_batch=100):
    bat_per_epo = int(dataset1.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):

        for j in range(bat_per_epo):
            d_loss1_I, d_loss2_I, g_loss_I = 0, 0, 0
            # train D1
            # get 'real' samples
            X_real, y_real = generate_real_samples(dataset1, half_batch)
            d_loss1_I, _ = d_model_1.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake_1, y_fake_1 = generate_fake_samples_1(
                g_model_1, latent_dim, half_batch)
            d_loss2_I, _ = d_model_1.train_on_batch(X_fake_1, y_fake_1)

            # train G1
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss_I = gan_model_1.train_on_batch(X_gan, y_gan)

            d_loss1_II, d_loss2_II, g_loss_II = 0, 0, 0
            # train D2
            # train on real
            X_real, y_real = generate_real_samples(dataset2, half_batch)
            d_loss1_II, _ = d_model_2.train_on_batch(X_real, y_real)
            # train on fake

            X_fake_2, y_fake = generate_fake_samples_2(
                g_model_2, X_fake_1, latent_dim, half_batch)
            d_loss2_II, _ = d_model_2.train_on_batch(X_fake_2, y_fake)

            # train G2
            X_fake_1, _ = generate_fake_samples_1(
                g_model_1, latent_dim, n_batch)
            noise = generate_latent_points(latent_dim, n_batch)
            X_gan = [X_fake_1,  noise]
            y_gan = ones((n_batch, 1))
            g_loss_II = gan_model_2.train_on_batch(X_gan, y_gan)

            # summarize loss on this batch
            print('>%d, %d/%d, d1_I=%.3f, d2_I=%.3f g_I=%.3f, d1_II=%.3f, d2_II=%.3f g_II=%.3f' %
                  (i+1, j+1, bat_per_epo, d_loss1_I, d_loss2_I, g_loss_I, d_loss1_II, d_loss2_II, g_loss_II))
        
        if (i+1) % 10 == 0:
            summarize_performance(
                i, g_model_1, d_model_1, g_model_2, d_model_2, dataset1, dataset2, latent_dim)


# size of the latent space
latent_dim = 100

d_model_1 = define_discriminator_1()
# d_model_1 = load_model('dis1_model2000.h5')
g_model_1 = define_generator_1(latent_dim)
# g_model_1 = load_model('gen1_model2000.h5')

gan_model_1 = define_gan_1(g_model_1, d_model_1)

d_model_2 = define_discriminator_2()
# d_model_2 = load_model('dis2_train_model650.h5')
g_model_2 = define_generator_2(latent_dim)
# g_model_2 = load_model('gen2_train_model650.h5')

gan_model_2 = define_gan_2(g_model_2, d_model_2)



dataset32, dataset64, dataset128 = load_real_samples()

n_epochs = 1000
n_batch = 100
train(g_model_1, d_model_1, g_model_2, d_model_2, gan_model_1, gan_model_2, dataset32, dataset64,
      latent_dim, n_epochs=n_epochs, n_batch=n_batch)
