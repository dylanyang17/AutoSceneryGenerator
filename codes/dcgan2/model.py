import tensorflow as tf
import numpy as np
import os
import math
from six.moves import xrange
import time
from PIL import Image
from plot import merge
from matplotlib import pyplot as plt
import scipy


def generate_latent_points(latent_dim, n_samples):
    x = np.random.uniform(-1, 1, (n_samples, latent_dim))
    return x


class DCGAN(object):
    def __init__(self,
                 sess,
                 input_height,
                 input_width,
                 batch_size,
                 sample_num,
                 output_height,
                 output_width,
                 z_dim,
                 dataset_name,
                 max_to_keep,
                 checkpoint_dir,
                 sample_dir,
                 out_dir,
                 data_dir,
                 g_last_channel_count=64,
                 d_first_channel_count=64,
                 color_channel=3):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.g_last_channel_count = g_last_channel_count
        self.d_first_channel_count = d_first_channel_count
        self.color_channel = color_channel
        self.batch_size = batch_size
        self.sample_num = 64
        self.z_dim = z_dim
        self.dataset_name = dataset_name
        self.max_to_keep = max_to_keep
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.out_dir = out_dir
        self.data_dir = data_dir
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.build_model()

    def build_model(self):
        image_dims = [self.output_height,
                      self.output_width, self.color_channel]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images_input')
        real_inputs = self.inputs
        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.G = self.generator(self.z)
        self.sampler = self.generator(self.z, is_train=False)
        self.D_real, self.D_real_logits = self.discriminator(
            real_inputs, reuse=False)
        self.D_fake, self.D_fake_logits = self.discriminator(
            self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        train_vars = tf.trainable_variables()
        self.d_vars = [var for var in train_vars if 'd_' in var.name]
        self.g_vars = [var for var in train_vars if 'g_' in var.name]
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

    def prepare_data(self, config):

        print('Preparing data...')
        self.path = os.path.join(
            config.data_dir, config.dataset)
        self.data = []
        for filename in os.listdir(self.path):
            try:
                filePath = os.path.join(self.path, filename)
                with Image.open(filePath) as img:
                    if(self.output_height == self.input_height and self.output_width == self.input_width):
                        pass
                    else:
                        img = img.resize(
                            (self.output_height, self.output_width))
                        img = np.array(img)
                    if img.shape == (self.output_height,  self.output_width, 3):
                        self.data.append(img)
            except:
                print('bad image: {}'.format(filename))
        self.data = np.array(self.data)
        self.data = self.data.astype('float32')
        self.data = self.data/127.5 - 1
        print('{} data images'.format(len(self.data)))

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) .minimize(
            self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate,
                                         beta1=config.beta1) .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()
        sample_z = generate_latent_points(
            self.z_dim, self.sample_num)
        self.prepare_data(config=config)

        sample_data = self.data[0:self.sample_num]
        counter = 1
        start_time = time.time()
        for epoch in xrange(config.epoch):
            np.random.shuffle(self.data)
            batch_idxs = len(self.data) // config.batch_size
            for idx in xrange(0, int(batch_idxs)):
                batch_z = generate_latent_points(self.z_dim,
                                                 config.batch_size).astype(np.float32)

                batch_images = self.data[idx *
                                         config.batch_size:(idx+1)*config.batch_size]
                # Update D network
                _ = self.sess.run([d_optim],
                                  feed_dict={self.inputs: batch_images, self.z: batch_z})

                # Update G network
                _ = self.sess.run([g_optim],
                                  feed_dict={self.z: batch_z})
                _ = self.sess.run([g_optim],
                                  feed_dict={self.z: batch_z})

                d_loss_fake_eval = self.d_loss_fake.eval({self.z: batch_z})
                d_loss_real_eval = self.d_loss_real.eval(
                    {self.inputs: batch_images})
                g_loss_eval = self.g_loss.eval({self.z: batch_z})

                print("[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (counter, epoch, config.epoch, idx, batch_idxs,
                         time.time() - start_time, d_loss_fake_eval+d_loss_real_eval, g_loss_eval))
                if counter % config.sample_freq == 0:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_z,
                            self.inputs: sample_data,
                        },
                    )
                    save_images(samples,
                                './{}/train_{:08d}.png'.format(config.sample_dir, counter))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                          (d_loss, g_loss))
                if counter % config.ckpt_freq == 0:
                    self.save(config.checkpoint_dir, counter)

                counter += 1

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            out0 = lrelu(
                conv2d(image, self.d_first_channel_count, name='d_layer0_conv'))
            out1 = lrelu(self.d_bn1(
                conv2d(out0, self.d_first_channel_count*2, name='d_layer1_conv')))
            out2 = lrelu(self.d_bn2(
                conv2d(out1, self.d_first_channel_count*4, name='d_layer2_conv')))
            out3 = lrelu(self.d_bn3(
                conv2d(out2, self.d_first_channel_count*8, name='d_layer3_conv')))
            out4, _, _ = linear(tf.reshape(
                out3, [self.batch_size, -1]), 1, 'd_layer4_linear')

            return tf.nn.sigmoid(out4), out4

    def generator(self, z, is_train=True):
        with tf.variable_scope("generator") as scope:
            if not is_train:
                scope.reuse_variables()
            s_h16, s_w16 = self.output_height, self.output_width
            s_h8, s_w8 = conv_out_size_with_same_padding(
                s_h16, 2), conv_out_size_with_same_padding(s_w16, 2)
            s_h4, s_w4 = conv_out_size_with_same_padding(
                s_h8, 2), conv_out_size_with_same_padding(s_w8, 2)
            s_h2, s_w2 = conv_out_size_with_same_padding(
                s_h4, 2), conv_out_size_with_same_padding(s_w4, 2)
            s_h, s_w = conv_out_size_with_same_padding(
                s_h2, 2), conv_out_size_with_same_padding(s_w2, 2)

            self.z_, self.layer0_weight, self.layer0_bias = linear(
                z, self.g_last_channel_count*8*s_h*s_w, 'g_layer0_linear')
            self.out0 = tf.reshape(
                self.z_, [-1, s_h, s_w, self.g_last_channel_count * 8])
            out0 = tf.nn.relu(self.g_bn0(self.out0, train=is_train))

            self.out1, self.layer1_weight, self.layer1_bias = conv2d_transpose(
                out0, [self.batch_size, s_h2, s_w2, self.g_last_channel_count*4], name='g_layer1', )
            out1 = tf.nn.relu(self.g_bn1(self.out1, train=is_train))

            out2, self.layer2_weight, self.layer2_bias = conv2d_transpose(
                out1, [self.batch_size, s_h4, s_w4, self.g_last_channel_count*2], name='g_layer2')
            out2 = tf.nn.relu(self.g_bn2(out2, train=is_train))

            out3, self.layer3_weight, self.layer3_bias = conv2d_transpose(
                out2, [self.batch_size, s_h8, s_w8, self.g_last_channel_count*1], name='g_layer3')
            out3 = tf.nn.relu(self.g_bn3(out3, train=is_train))

            out4, self.layer4_weight, self.layer4_bias = conv2d_transpose(
                out3, [self.batch_size, s_h16, s_w16, self.color_channel], name='g_layer4')

        return tf.nn.tanh(out4)

    def save(self, checkpoint_dir, step, filename='model', ckpt=True):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if ckpt:
            self.saver.save(self.sess,
                            os.path.join(checkpoint_dir, filename),
                            global_step=step)

    def load(self, checkpoint_dir):
        pass


def conv2d(inputx, filters, kernel_height=5, kernel_width=5, stride_h=2, stride_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, inputx.get_shape()[-1], filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(inputx, w, strides=[
                            1, stride_h, stride_w, 1], padding='SAME')
        biases = tf.get_variable(
            'biases', [filters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv


def conv2d_transpose(inputx, output_shape,
                     kernel_height=5, kernel_width=5, stride_h=2, stride_w=2, stddev=0.02,
                     name="conv2d_transpose"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_height, kernel_width, output_shape[-1], inputx.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(inputx, w, output_shape=output_shape,
                                        strides=[1, stride_h, stride_w, 1])
        biases = tf.get_variable(
            'biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        return deconv, w, biases


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def save_images(images, image_path):
    num_images = images.shape[0]
    h = int(np.floor(np.sqrt(num_images)))
    w = int(np.ceil(np.sqrt(num_images)))
    images = (images + 1) / 2.0

    image_combined = merge(images, (h, w))
    return scipy.misc.imsave(image_path, image_combined)


def linear(inputx, output_size, scope="Linear", stddev=0.02, bias_init=0.0):
    shape = inputx.get_shape()
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_init))
        return tf.matmul(inputx, w) + bias, w, bias


def conv_out_size_with_same_padding(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)
