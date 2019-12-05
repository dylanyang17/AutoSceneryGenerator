from  dataset import getDatasets
import tensorflow as tf
import matplotlib.pyplot as plt
import os
gpu_options = tf.GPUOptions(allow_growth=False)
lr = 1e-3
width = 128
height= 128
channel = 3
EPOCH = 300
BatchSize = 10
checkpoint_dir = './checkpoint'
def dense_layer(incoming, unitsCount):
    fc = tf.layers.dense(
        incoming, kernel_initializer=tf.contrib.layers.xavier_initializer(), units=unitsCount)
    return fc

def build_conv_net():
    x_input = tf.placeholder(tf.float32, [None, 128,128,3])
    net = tf.layers.conv2d(x_input, filters=64,kernel_size= [3, 3],padding='SAME',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2],strides=2)
    print(net.shape)
    net = tf.layers.conv2d(net, filters=128,kernel_size= [5, 5],padding='SAME',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2],strides=2)
    print(net.shape)
    net = tf.layers.conv2d(net, filters=256,kernel_size= [5, 5],padding='SAME',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2],strides=2)
    print(net.shape)
    # net = tf.layers.conv2d(net, filters=256,kernel_size= [5, 5],padding='SAME',activation=tf.nn.relu)
    # net = tf.layers.max_pooling2d(net, pool_size=[2,2],strides=2)
    # print(net.shape)
    net = tf.layers.conv2d(net, filters=256,kernel_size= [3, 3],padding='SAME',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2],strides=2)
    print(net.shape)
    net = tf.layers.conv2d(net, filters=256,kernel_size= [2, 2],padding='SAME',activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, pool_size=[2,2],strides=2)
    print(net.shape)

    net = tf.layers.conv2d_transpose(net, 128, kernel_size=(4,4),strides=(4,4), padding='valid', activation=tf.nn.relu)
    print(net.shape)
    net = tf.layers.conv2d_transpose(net, 64, kernel_size=(2,2), strides=(2,2),padding='valid', activation=tf.nn.relu)
    print(net.shape)
    net = tf.layers.conv2d_transpose(net, 32, kernel_size=(2,2), strides=(2,2),padding='valid', activation=tf.nn.relu)
    print(net.shape)
    # net = tf.layers.conv2d_transpose(net, 32, kernel_size=(2,2), strides=(2,2),padding='SAME', activation=tf.nn.relu)
    # print(net.shape)
    net = tf.layers.conv2d_transpose(net, 128, kernel_size=(2,2), strides=(2,2),padding='SAME', activation=tf.nn.relu)
    print(net.shape)
    net = tf.layers.conv2d_transpose(net, 3, kernel_size=(3,3), strides=(1,1),padding='SAME', activation=tf.nn.relu)
    print(net.shape)

    
    diff = tf.square(net-x_input)
    loss = tf.reduce_mean(diff)
    return x_input, loss, net


data = getDatasets()
data = data[:4000]
test_batch = data[:100]
x_input, loss, net = build_conv_net()
global_step = tf.Variable(0, trainable=False)
params = tf.trainable_variables()
train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step, var_list=params)
saver = tf.train.Saver(max_to_keep=300)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    

    if tf.train.get_checkpoint_state(checkpoint_dir):
        print('Load model')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
        print('Model restored')
    else:
        print('New model')
        tf.global_variables_initializer().run()

    for j in range(20):
        plt.imsave(os.path.join('./output/'+str(j), 'target.jpg'), test_batch[j]/255.0)

    for i in range(EPOCH):
        loss_value = 0.0
        st, ed, times = 0, BatchSize, 0
        while st < len(data) and ed <= len(data):
            X_batch = data[st:ed]
            feed = {x_input: X_batch}
            loss_, net_, _ = sess.run([loss, net, train_op], feed)
            loss_value += loss_

            st, ed = ed, ed+BatchSize
            times += 1
        loss_value /= times

        print("Epoch {0}: loss {1}".format(i, loss_value))
        saver.save(sess, os.path.join(checkpoint_dir, 'model0'),global_step=global_step)
        
        # fig = plt.figure()
        test_loss, test_out = sess.run([loss, net], {x_input: test_batch})
        for j in range(20):
            test_out[j][test_out[j]>255]=255
            # print(test_out[j].shape)
            plt.imsave(os.path.join('./output/'+ str(j), str(global_step.eval())+'.jpg'), test_out[j]/255.0)
    # plt.show()

