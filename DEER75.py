
#==========================================
# Title:  DEER network
# Author: Huidong Xie
# Biomedical Engineering Undergraduate student at Rensselaer Polytechnic Institute
# Python 3.6, Tensorflow 1.11
#==========================================


import numpy as np
import h5py
from sklearn.utils import shuffle
import math
from tensorflow.python.client import device_lib
import tensorflow as tf
import random

gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options, allow_soft_placement=True)
sess = tf.Session(config=session_config)

print(device_lib.list_local_devices())


def deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


############################
pi_on_180 = 0.017453292519943295
pi = tf.constant(math.pi, tf.float32)
LEARNING_RATE = 1e-4  # keep decreasing while training
batch_size = 3
start_epoch = 0
num_epoch = 100
width = 1024  # width of image
views = 150
theta = np.linspace(0, 360, views, endpoint=False)
print(theta)
T = tf.convert_to_tensor(theta)
T = tf.cast(T, tf.float32)
disc_iters = 4
lamda_P = 0  # hyperparameters
lamda_R = 0  # hyperparameters
C = 1
############################


def two_dim_conv(batch, name, n_filter):
    c = tf.layers.conv2d(
        inputs=batch,
        filters=n_filter,
        kernel_size=[3, 3],
        strides=1,
        padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        use_bias=True,
        activation=None,
        name=name,
        reuse=tf.AUTO_REUSE
    )
    # print("{}:{}".format(name, c.get_shape()))
    return c


def generator1(batch, batch_size, PI):
    with tf.device('/device:GPU:0'):
        g5 = tf.zeros([batch_size, width, width, 1], dtype=tf.float32)

        W = tf.get_variable(name="weights0", shape=[views, C, width, width], dtype=tf.float32,
                            initializer=tf.constant_initializer(1 / views / C), trainable=True)
        i = 0

        def body(g5_e, i, batch, W):
            projection = tf.expand_dims(tf.gather(batch, i, axis=1), 1)
            print(projection)

            def fc(x, W):
                x = tf.squeeze(x, -1)
                print(x)
                planes_i = tf.expand_dims(tf.multiply(x, W[0]), -1)
                for j in range(1, C):
                    planes_sub = tf.expand_dims(tf.multiply(x, W[j]), -1)
                    planes_i = planes_i + planes_sub

                planes_i = tf.reshape(planes_i, [1, width, width, 1])
                return planes_i

            planes = fc(projection[0], W[i])
            for k in range(1, batch_size):
                planes_next = fc(projection[k], W[i])
                planes = tf.concat([planes, planes_next], 0)
            planes = tf.reshape(planes, [batch_size, width, width, 1])
            angle = T[i] * (pi / 180)
            planes = tf.contrib.image.rotate(planes, angles=angle, interpolation='BILINEAR')
            g5_e = g5_e + planes
            return g5_e, i + 1, batch, W

        def condition(g5_e, i, batch, W):
            return tf.less(i, views)

        g5, _, _, _ = tf.while_loop(condition, body, loop_vars=[g5, i, batch, W], parallel_iterations=PI)
        return g5


def generator2(g5, FBP_img):
    g5 = tf.concat([g5, FBP_img], -1)
    ug1_0 = tf.layers.conv2d(inputs=g5, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g1_1',
                             activation=tf.nn.relu)
    # ug1 = dense_block(ug1_0, 'ug1_0', 32)

    ug2_0 = tf.layers.conv2d(inputs=ug1_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g2_1',
                             activation=tf.nn.relu)
    # ug2 = dense_block(ug2_0, 'ug2_0', 32)

    ug3_0 = tf.layers.conv2d(inputs=ug2_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g3_1',
                             activation=tf.nn.relu)

    ug4_0 = tf.layers.conv2d(inputs=ug3_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g4_1',
                             activation=tf.nn.relu)

    ug5_0 = tf.layers.conv2d_transpose(inputs=ug4_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g5_1',
                                       activation=tf.nn.relu)
    ug5_0 = tf.concat([ug3_0, ug5_0], -1)

    ug6_0 = tf.layers.conv2d_transpose(inputs=ug5_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g6_1',
                                       activation=tf.nn.relu)
    ug6_0 = tf.concat([ug2_0, ug6_0], -1)

    ug7_0 = tf.layers.conv2d_transpose(inputs=ug6_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g7_1',
                                       activation=tf.nn.relu)
    ug7_0 = tf.concat([ug1_0, ug7_0], -1)
    # ug4 = dense_block(ug4_0, 'ug4_0', 32)
    ug8_0 = tf.layers.conv2d_transpose(inputs=ug7_0, filters=32, kernel_size=[5, 5], strides=(1, 1), padding='valid',
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g8_1',
                                       activation=tf.nn.relu)
    output = two_dim_conv(ug8_0, 'output1', 1)

    return tf.nn.relu(output)


def discriminator(batch):
    # First convolutional and pool layers
    # This finds 32 different 2 x 2 pixel features
    with tf.device('/device:GPU:1'):
        d1 = tf.layers.conv2d(
            inputs=batch,
            filters=32,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_d1'
        )
        d1 = tf.nn.leaky_relu(d1, alpha=0.2)
        # d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("d1,{}".format(d1.get_shape()))

        d2 = tf.layers.conv2d(
            inputs=d1,
            filters=32,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_d2'
        )
        d2 = tf.nn.leaky_relu(d2, alpha=0.2)
        # d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("d2,{}".format(d2.get_shape()))

        d3 = tf.layers.conv2d(
            inputs=d2,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_d3'
        )
        d3 = tf.nn.leaky_relu(d3, alpha=0.2)
        # d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("d3,{}".format(d3.get_shape()))

        d4 = tf.layers.conv2d(
            inputs=d3,
            filters=64,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_d4'
        )
        d4 = tf.nn.leaky_relu(d4, alpha=0.2)
        # d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("d4,{}".format(d4.get_shape()))

        d5 = tf.layers.conv2d(
            inputs=d4,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_d5'
        )
        d5 = tf.nn.leaky_relu(d5, alpha=0.2)
        # d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("d5,{}".format(d5.get_shape()))

        d6 = tf.layers.conv2d(
            inputs=d5,
            filters=128,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv_d6'
        )
        d6 = tf.nn.leaky_relu(d6, alpha=0.2)
        # d4 = tf.nn.avg_pool(d4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print("d6,{}".format(d6.get_shape()))

        d7 = tf.layers.dense(
            inputs=d6,
            units=1024,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0),
            name='conv_d7'
        )
        d7 = tf.nn.leaky_relu(d7)
        print("d7:{}".format(d7.get_shape()))

        d8 = tf.layers.dense(
            inputs=d7,
            units=1,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.constant_initializer(0),
            name='conv_d8'
        )
        print("d8:{}".format(d8.get_shape()))

        # d8 contains unscaled values
    return d8


if __name__ == '__main__':
    lr = tf.placeholder(dtype=tf.float32, shape=[])  # learning rate
    FD_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, width, width, 1], name='FD_placeholder')
    projection_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, views, width, 1],
                                       name='projection_holder')

    projection_holder_test = tf.placeholder(dtype=tf.float32, shape=[1, views, width, 1],
                                            name='projection_holder_test_o')
    FD_placeholder_test = tf.placeholder(dtype=tf.float32, shape=[1, width, width, 1], name='FD_placeholder_test')

    FBP_img_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, width, width, 1], name='FBP_placeholder')
    FBP_img_placeholder_test = tf.placeholder(dtype=tf.float32, shape=[1, width, width, 1], name='FBP_placeholder_test')

    with tf.variable_scope('generator_model1', reuse=tf.AUTO_REUSE) as scope_generator_model1:
        Gz1 = generator1(projection_holder, batch_size=batch_size, PI=5)
        Gz_test1 = generator1(projection_holder_test, batch_size=1, PI=views)

    with tf.variable_scope('generator_model2', reuse=tf.AUTO_REUSE) as scope_generator_model2:
        Gz2 = generator2(Gz1, FBP_img_placeholder)
        Gz_test2 = generator2(Gz_test1, FBP_img_placeholder_test)

    alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
    with tf.variable_scope('discriminator_model') as scope_discriminator_model:
        Dx = discriminator(FD_placeholder)
        scope_discriminator_model.reuse_variables()
        Dg1 = discriminator(Gz1)
        interpolates1 = alpha * tf.reshape(FD_placeholder, [batch_size, -1]) + (1 - alpha) * \
                        tf.reshape(Gz1, [batch_size, -1])
        interpolates1 = tf.reshape(interpolates1, [batch_size, 1024, 1024, 1])
        gradients1 = tf.gradients(discriminator(interpolates1), [interpolates1])[0]

    slopes1 = tf.sqrt(tf.reduce_sum(tf.square(gradients1), axis=1))
    gradient_penalty1 = tf.reduce_mean((slopes1 - 1.) ** 2)  # p_loss
    difference1 = tf.reduce_mean(Dg1) - tf.reduce_mean(Dx)  # w distance
    disc_loss1 = difference1 + 10 * gradient_penalty1
    gen_cost1 = -tf.reduce_mean(Dg1)
    disc_loss = disc_loss1

    mae = tf.reduce_mean(tf.abs(Gz2 - FD_placeholder))
    mae_mid = tf.reduce_mean(tf.abs(Gz1 - FD_placeholder))
    ssim1 = tf.squeeze(tf.reduce_mean(tf.image.ssim(Gz2, FD_placeholder, 1.0)))
    ssim_loss1 = 1 - ssim1
    g_loss2 = ssim_loss1 * lamda_P + mae + lamda_R * gen_cost1
    g_loss1 = mae_mid

    gen_variables1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_model1')
    gen_variables2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_model2')
    disc_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_model')

    # Train the generator
    g_trainer1 = tf.train.AdamOptimizer(learning_rate=lr / 10.0).minimize(g_loss1, var_list=gen_variables1)
    g_trainer2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss2, var_list=gen_variables2)
    with tf.device('/device:GPU:1'):
        d_trainer = tf.train.AdamOptimizer(learning_rate=lr * 5.0).minimize(disc_loss, var_list=disc_variables)
    #####################################################

    Patient_IDs = [] # Training patient IDs

    saver = tf.train.Saver()
    ###############################################################
    dataset_index = 1
    print("enter Session")
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, './FBP_50_imagenet/model.ckpt')
    print('model restored')
    print("Begin training")
    val_lr = LEARNING_RATE
    for iteration in range(start_epoch, num_epoch):
        random.shuffle(Patient_IDs)

        val_lr = LEARNING_RATE / np.sqrt(iteration + 1)
        count = 0
        for ID in Patient_IDs:
            # break
            print(ID)
            filename = "/data/HUIDONG/Koning_FDK/" + str(ID) + ".h5"
            f = h5py.File(filename, 'r')
            label = np.array(f['image'], dtype=np.float32)
            f.close()
            filename = "/data/HUIDONG/Koning_FDK_150_views_processed/" + str(ID) + ".h5"
            f = h5py.File(filename, 'r')
            filtered_sino = np.array(f['filtered_sino'], dtype=np.float32)
            fbp_img = np.array(f['fbp_image'], dtype=np.float32)
            f.close()

            label, filtered_sino, fbp_img = shuffle(label, filtered_sino, fbp_img)
            num_batches = label.shape[0] // batch_size
            print("###################################")
            print("epoch:{}".format(iteration))
            print("learning rate:{}".format(val_lr))
            print("num batch:{}".format(num_batches))
            print("###################################")

            for i in range(num_batches):
                print("batch:{}".format(i))

                for _ in range(disc_iters):
                    idx = np.random.permutation(label.shape[0])
                    batch_label = label[idx[:batch_size]]
                    feed_img = np.squeeze(batch_label)
                    feed_img = np.expand_dims(feed_img, -1)
                    feed_projection = filtered_sino[idx[:batch_size]]
                    feed_projection = np.squeeze(feed_projection)
                    feed_projection = np.expand_dims(feed_projection, -1)
                    feed_fbp_img = fbp_img[idx[:batch_size]]
                    feed_fbp_img = np.squeeze(feed_fbp_img)
                    feed_fbp_img = np.expand_dims(feed_fbp_img, -1)
                    _ = sess.run(d_trainer, feed_dict={FD_placeholder: feed_img, projection_holder: feed_projection,
                                                       lr: val_lr, FBP_img_placeholder: feed_fbp_img})

                batch_label = label[i * batch_size: (i + 1) * batch_size]
                feed_img = np.squeeze(batch_label)
                feed_img = np.expand_dims(feed_img, -1)
                feed_projection = filtered_sino[i * batch_size: (i + 1) * batch_size]
                feed_projection = np.squeeze(feed_projection)
                feed_projection = np.expand_dims(feed_projection, -1)
                feed_fbp_img = fbp_img[i * batch_size: (i + 1) * batch_size]
                feed_fbp_img = np.squeeze(feed_fbp_img)
                feed_fbp_img = np.expand_dims(feed_fbp_img, -1)

                _, _, _mae, _mae_mid, _ssim, _gen_cost, _difference = sess.run(
                    [g_trainer1, g_trainer2, mae, mae_mid, ssim1, gen_cost1, difference1],
                    feed_dict={FD_placeholder: feed_img,
                               projection_holder: feed_projection, FBP_img_placeholder: feed_fbp_img,
                               lr: val_lr})

                print("Epoch:{} {} {} mae:{} mae_mid:{} ssim:{} gen_cost:{} differencee:{} ".format(iteration, ID, i,
                                                                                                    _mae, _mae_mid,
                                                                                                    _ssim,
                                                                                                    _gen_cost,
                                                                                                    _difference))

        saver.save(sess, './FBP_50_imagenet/model.ckpt')
