import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from dataset import getDatasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data = getDatasets()
num_steps = 10000
batch_size = 100
lr_generator = 0.002
lr_discriminator = 0.002


image_dim = 128*128
noise_dim = 100

noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])

is_training = tf.placeholder(tf.bool)


def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)


def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=16 * 16 * 256)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.reshape(x, shape=[-1, 16, 16, 256])
        x = tf.layers.conv2d_transpose(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        print(x.shape)
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        print(x.shape)
        x = tf.layers.conv2d_transpose(x, 3, 5, strides=2, padding='same')
        x = tf.nn.tanh(x)
        print(x.shape)
        return x


def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.reshape(x, shape=[-1, 32*32*128])
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyrelu(x)
        x = tf.layers.dense(x, 2)
    return x


gen_sample = generator(noise_input)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
stacked_gan = discriminator(gen_sample, reuse=True)

disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))

disc_loss = disc_loss_real + disc_loss_fake

gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))


optimizer_gen = tf.train.AdamOptimizer(
    learning_rate=lr_generator, beta1=0.5, beta2=0.999)
optimizer_disc = tf.train.AdamOptimizer(
    learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)


gen_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

disc_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')


gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')

global_step = tf.Variable(0, trainable=False)
with tf.control_dependencies(gen_update_ops):
    train_gen = optimizer_gen.minimize(
        gen_loss, var_list=gen_vars, global_step=global_step)
disc_update_ops = tf.get_collection(
    tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)


init = tf.global_variables_initializer()

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=300)
checkpoint_dir = './gan'
if tf.train.get_checkpoint_state(checkpoint_dir):
    print('Load model')
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    print('Model restored')
else:
    print('New model')
    sess.run(init)

st, ed, times = 0, batch_size, 0
for i in range(1, num_steps+1):

    if(ed >= 5780):
        st, ed, times = 0, batch_size, 0
    batch_x = data[st:ed]
    st, ed = ed, ed+batch_size
    times += 1
    batch_x = np.reshape(batch_x, newshape=[-1, 128, 128, 3])

    batch_x = batch_x * 2. - 1.

    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, dl = sess.run([train_disc, disc_loss], feed_dict={
                     real_image_input: batch_x, noise_input: z, is_training: True})

    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, gl, genPic = sess.run([train_gen, gen_loss, gen_sample], feed_dict={
                             noise_input: z, is_training: True})
    step_eval = global_step.eval(sess)
    if i % 10 == 0:
        plt.imsave(os.path.join('./output', 'dcgan',
                                str(step_eval)+'.jpg'), genPic[0])
    if i % 10 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' %
              (step_eval, gl, dl))
        if i % 500 == 0:
            saver.save(sess, os.path.join(checkpoint_dir,
                                          'model0'), global_step=global_step)
