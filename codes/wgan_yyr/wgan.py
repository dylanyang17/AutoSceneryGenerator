import sys

import numpy as np
from functools import reduce
import os
import datetime
import matplotlib.pyplot as plt
import keras

log_path = './log.txt'
train_dir = './train'


def debug(s):
    # 输出字符串, 并且追加打印到log.txt中
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:: ')
    print(timestamp + s)
    with open(file=log_path, mode='a') as f:
        print(timestamp + s, file=f)

class WGAN():
    def __init__(self):
        self.bn_momentum = 0.95
        self.channels = 3
        self.img_shape = (64, 64, self.channels)
        self.noise_shape = (100,)

        self.d_train_times = 5
        self.clip_value = 0.01
        self.optimizer = keras.optimizers.RMSprop(lr=0.00005)
        self.data_path = '../../data/mountains64.npy'

        self.base_discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.discriminator = keras.models.Model(self.base_discriminator.inputs, self.base_discriminator.outputs)
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.discriminator_frozen = keras.models.Model(self.base_discriminator.inputs, self.base_discriminator.outputs)
        self.discriminator_frozen.trainable = False

        z = keras.layers.Input(shape=self.noise_shape)
        img = self.generator(z)
        score = self.discriminator_frozen(img)
        self.combined = keras.models.Model(z, score)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=self.optimizer,
                                metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return keras.backend.mean(y_true * y_pred)

    def build_discriminator(self):
        """
        构建 Discriminator，注意传入值为 -1~1 的图片
        :return:
        """
        model = keras.models.Sequential()
        # model.add(keras.layers.Conv2D(input_shape=self.img_shape, filters=64, kernel_size=5, strides=2, padding='same'))
        # model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        # model.add(keras.layers.LeakyReLU())
        # model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=512, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Conv2D(filters=1024, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1))  # 不加Activation

        img = keras.layers.Input(shape=self.img_shape)
        score = model(img)
        return keras.models.Model(img, score)

    def build_generator(self):
        """
        构建 Generator，注意生成值为 -1~1 的图片
        :return:
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(1024*4*4, input_shape=self.noise_shape))
        model.add(keras.layers.Reshape((4, 4, 1024)))
        model.add(keras.layers.Deconv2D(filters=1024, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Deconv2D(filters=512, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Deconv2D(filters=256, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Activation('relu'))
        # model.add(keras.layers.Deconv2D(filters=128, kernel_size=5, strides=2, padding='same'))
        # model.add(keras.layers.BatchNormalization(momentum=self.bn_momentum))
        model.add(keras.layers.Deconv2D(filters=self.channels, kernel_size=5, strides=2, padding='same'))
        model.add(keras.layers.Activation('tanh'))
        model.add(keras.layers.Reshape(self.img_shape))

        z = keras.layers.Input(shape=self.noise_shape)
        img = model(z)
        return keras.models.Model(z, img)

    def train(self, start_epoch, end_epoch, batch_size, save_interval):
        """
        进行训练，训练的起始轮数为 start_epoch，终止轮数为 end_epoch（闭区间），
        每一轮选出 batch_size 张图片，每隔 save_interval 轮进行一次保存，注意每一轮中 discriminator 训练 d_train_times 次,
        而 generator 训练 1 次
        """
        # data 的形状: (number of images, 256, 256, 3)，注意将其变换到 [-1, 1] 上
        data = np.load(self.data_path)
        data = data / 127.5 - 1
        real = -np.ones(shape=(batch_size, 1))
        fake = np.ones(shape=(batch_size, 1))
        d_loss_list = []
        g_loss_list = []
        for epoch in range(start_epoch + 1, end_epoch + 1):
            start_time = datetime.datetime.now()
            debug('training on epoch ' + str(epoch))
            for i in range(self.d_train_times):
                # Train Discriminator
                idx = np.random.randint(0, data.shape[0], batch_size)
                real_imgs = data[idx] + np.random.normal(0, 0.0001, self.img_shape)  # 加入噪声
                z = np.random.normal(0, 1, size=(batch_size,)+self.noise_shape)
                fake_imgs = self.generator.predict(z)
                d_loss_real = self.discriminator.train_on_batch(real_imgs, real)
                d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                debug('D:: d_train iteration: %d  d_loss: %f  d_acc: %f' % (i, d_loss[0], d_loss[1]))
                d_loss_list.append(d_loss)

                # Clip
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # Train Generator
            g_loss = self.combined.train_on_batch(z, real)
            g_loss_list.append(g_loss)
            debug('G:: g_loss: %f  g_acc: %f]' %
              (g_loss[0], g_loss[1]))

            time_cost = (datetime.datetime.now()-start_time).total_seconds()
            debug('finish epoch %d  time_cost: %d' % (epoch, time_cost))

            # Save
            if epoch % save_interval == 0:
                self.save_model(epoch)
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        """
        保存第 epoch 轮生成的图片
        :param epoch: 要保存信息的轮数
        :return:
        """
        debug('save images, epoch: %d' % epoch)
        save_dir = os.path.join(train_dir, 'epoch' + str(epoch))
        os.makedirs(save_dir, exist_ok=True)

        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c,)+self.noise_shape)
        fake_imgs = self.generator.predict(noise)

        # 变换到 [0, 255] 上
        fake_imgs = np.round((fake_imgs + 1) / 2 * 255).astype('uint8')

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(fake_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(save_dir, 'img.png'))
        plt.close()

    def choose_best_generated_images(self, sample_num, top_num):
        """
        采样 sample_num 张图片, 并且返回最好的 top_num 张
        :return: numpy数组, 形状为 (top_num, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        """
        sample_list = []
        z = np.random.normal(0, 1, size=(sample_num,)+self.noise_shape)
        img = np.array(self.generator.predict(z))
        score = np.array(self.discriminator.predict(img))
        for i in range(len(img)):
            sample_list.append([img[i], score[i]])
        # sample_list.sort(key=lambda x: abs(x[1] + 1))
        sample_list.sort(key=lambda x: x[1])
        sample_list = sample_list[:top_num]
        for i in range(len(sample_list)):
            print(sample_list[i][1])
            sample_list[i] = sample_list[i][0]
        return np.array(sample_list)

    def save_model(self, epoch):
        """
        保存第 epoch 轮的信息
        :param epoch: 要保存信息的轮数
        :return:
        """
        # 保存模型
        debug('save model, epoch: %d' % epoch)
        save_dir = os.path.join(train_dir, 'epoch' + str(epoch))
        os.makedirs(save_dir, exist_ok=True)

        # self.combined.save(os.path.join(save_dir, 'combined'))
        self.generator.save(os.path.join(save_dir, 'generator'))
        self.base_discriminator.save(os.path.join(save_dir, 'base_discriminator'))

    def load_model(self, epoch):
        """
        载入第 epoch 轮保存的信息
        :param epoch: 保存信息的轮数
        :return:
        """
        debug('load model, epoch: %d' % epoch)
        save_dir = os.path.join(train_dir, 'epoch' + str(epoch))
        # self.combined = keras.models.load_model(os.path.join(save_dir, 'combined'))
        self.generator = keras.models.load_model(os.path.join(save_dir, 'generator'))
        self.base_discriminator = keras.models.load_model(os.path.join(save_dir, 'base_discriminator'))
        self.discriminator = keras.models.Model(self.base_discriminator.inputs, self.base_discriminator.outputs)
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.discriminator_frozen = keras.models.Model(self.base_discriminator.inputs, self.base_discriminator.outputs)
        self.discriminator_frozen.trainable = False

        z = keras.layers.Input(shape=self.noise_shape)
        img = self.generator(z)
        score = self.discriminator_frozen(img)
        self.combined = keras.models.Model(z, score)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=self.optimizer,
                              metrics=['accuracy'])

def filter_save_dir(s):
    """
    filter函数, 筛选出以epoch开头的目录, 均为保存数据的目录
    """
    return s.startswith('epoch')


def get_last_epoch():
    """
    获得最新保存的训练的 epoch
    :return: int 类型的 epoch，若不存在则返回 -1
    """
    dir_list = os.listdir(train_dir)
    dir_list = list(filter(filter_save_dir, dir_list))
    max_epoch = -1
    for name in dir_list:
        epoch = int(name[5:])
        max_epoch = max(epoch, max_epoch)
    return max_epoch

if __name__ == '__main__':
    wgan = WGAN()
    os.makedirs(train_dir, exist_ok=True)
    start_epoch = get_last_epoch()
    if start_epoch != -1:
        wgan.load_model(start_epoch)
    wgan.train(start_epoch=start_epoch, end_epoch=100000, batch_size=64, save_interval=50)
