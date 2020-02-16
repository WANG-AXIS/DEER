import os
import pydicom
import numpy as np
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math
from tensorflow.python.client import device_lib
from skimage.transform import radon, iradon
import tensorflow as tf
from skimage.measure import compare_ssim
import random
from skimage.transform import resize

dataset_path = '/data/HUIDONG/Koning_recon/'
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
start_epoch = 1
num_epoch = 10
width = 1024  # width of image
views = 75
theta = np.linspace(0, 360, views, endpoint=False)
print(theta)
T = tf.convert_to_tensor(theta)
T = tf.cast(T, tf.float32)
disc_iters = 4
lamda_P = 0.5 #0.6
lambda_vgg = 0.01  #0.01
lamda_R = 0.005  #0.0025
water = 0.22
air = 0.00027
C = 2

[X, Y] = np.mgrid[0:width, 0:width]
xpr = X - int(width) // 2
ypr = Y - int(width) // 2
radius = width // 2
reconstruction_circle_np = (xpr ** 2 + ypr ** 2) <= radius ** 2


############################


def write_dicom(ds, pixel_data, filename):
    if pixel_data.dtype != np.uint16:
        pixel_data = pixel_data.astype(np.uint16)
    ds.PixelData = pixel_data.tostring()
    ds.save_as(filename)
    return


def normalize(batch, lower=-200.0, upper=200.0):
    # batch = 1000 * ((batch - water) / (water - air))
    # batch = 600 * batch - 300
    # batch = (batch - lower) / (upper - lower)
    batch[batch > 1.0] = 1.0
    batch[batch < 0.0] = 0.0
    batch = 600 * batch - 300
    #[X, Y] = np.mgrid[0:1024, 0:1024]
    #xpr = X - int(1024) // 2
    #ypr = Y - int(1024) // 2
    #radius = 1024 // 2
    #reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius ** 2
    #batch[~reconstruction_circle] = 0.
    # batch = (batch - np.min(batch)) / (np.max(batch) - np.min(batch))
    return np.squeeze(batch)


def get_batch_projection(batch):
    batch = np.squeeze(batch)
    batch = np.transpose(batch)
    batch = np.expand_dims(batch, 0)
    batch = np.expand_dims(batch, -1)
    return batch


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


def dense_block(batch, name, n_out):
    conv_1 = two_dim_conv(batch, name + 'res_conv1_1', n_out)
    conv_1 = tf.nn.relu(conv_1)

    conv_2 = two_dim_conv(conv_1, name + 'res_conv2_1', n_out)
    conv_2 = tf.nn.relu(conv_2)
    conv_2 = tf.concat([conv_1, conv_2], -1)

    conv_3 = two_dim_conv(conv_2, name + 'res_conv3_1', n_out)
    conv_3 = tf.nn.relu(conv_3)

    return conv_3


def generator(batch, batch_size, PI):
    with tf.device('/device:GPU:0'):
        g5 = tf.zeros([batch_size, width, width, 1], dtype=tf.float32)

        W = tf.get_variable(name="weights0", shape=[views, C, width, 1], dtype=tf.float32,
                            initializer=tf.constant_initializer(1/views/C), trainable=True)
        W = W * tf.ones(shape=[1, 1, 1, width], dtype=tf.float32)
        B = tf.get_variable(name="bias0", shape=[views, C, width, 1], dtype=tf.float32,
                            initializer=tf.zeros_initializer(), trainable=True)
        i = 0

        def body(g5_e, i, batch, W, B):
            projection = tf.expand_dims(tf.gather(batch, i, axis=1), 1)
            print(projection)

            def fc(x, W, B):
                x = tf.squeeze(x, -1)
                print(x)
                planes_i = tf.expand_dims(tf.multiply(x, W[0]) + B[0], -1)
                for j in range(1, C):
                    planes_sub = tf.expand_dims(tf.multiply(x, W[j]) + B[j], -1)
                    planes_i = planes_i + planes_sub

                planes_i = tf.reshape(planes_i, [1, width, width, 1])
                return planes_i

            planes = fc(projection[0], W[i], B[i])
            for k in range(1, batch_size):
                planes_next = fc(projection[k], W[i], B[i])
                planes = tf.concat([planes, planes_next], 0)
            planes = tf.reshape(planes, [batch_size, width, width, 1])
            angle = T[i] * (pi / 180)
            planes = tf.contrib.image.rotate(planes, angles=angle, interpolation='BILINEAR')
            g5_e = g5_e + planes
            return g5_e, i + 1, batch, W, B

        def condition(g5_e, i, batch, W, B):
            return tf.less(i, views)

        g5, _, _, _, _ = tf.while_loop(condition, body, loop_vars=[g5, i, batch, W, B], parallel_iterations=PI)

        ug1_0 = tf.layers.conv2d(inputs=g5, filters=36, kernel_size=[3, 3], strides=(1, 1), padding='valid',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g1_1',
                                 activation=tf.nn.relu)
        ug1 = dense_block(ug1_0, 'ug1_0', 36)

        ug2_0 = tf.layers.conv2d(inputs=ug1, filters=36, kernel_size=[3, 3], strides=(1, 1), padding='valid',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g2_1',
                                 activation=tf.nn.relu)
        ug2 = dense_block(ug2_0, 'ug2_0', 36)

        ug3_0 = tf.layers.conv2d_transpose(inputs=ug2, filters=36, kernel_size=[3, 3], strides=(1, 1), padding='valid',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g3_1',
                                 activation=tf.nn.relu)
        ug3 = dense_block(ug3_0, 'ug3_0', 36)

        ug4_0 = tf.layers.conv2d_transpose(inputs=ug3, filters=36, kernel_size=[3, 3], strides=(1, 1), padding='valid',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='u_conv_g4_1',
                                 activation=tf.nn.relu)
        ug4 = dense_block(ug4_0, 'ug4_0', 36)

        output = two_dim_conv(ug4, 'output1', 1)

    return tf.clip_by_value(output, 0.0, 1.0)


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


def vgg_model(inputs):
    # outputs = tf.tile(inputs, (1,1,1,3)) * 255
    # img_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    # outputs = outputs - img_mean
    outputs = tf.concat([inputs * 255 - 103.939, inputs * 255 - 116.779, inputs * 255 - 123.68], 3)
    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv1_1')
    outputs = tf.layers.conv2d(outputs, 64, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv1_2')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool1')

    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv2_1')
    outputs = tf.layers.conv2d(outputs, 128, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv2_2')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool2')

    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv3_1')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv3_2')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv3_3')
    outputs = tf.layers.conv2d(outputs, 256, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv3_4')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool3')

    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv4_1')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv4_2')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv4_3')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv4_4')
    outputs = tf.layers.max_pooling2d(outputs, 2, strides=(2, 2), padding='same', name='pool4')

    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv5_1')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv5_2')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv5_3')
    outputs = tf.layers.conv2d(outputs, 512, 3, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu,
                               name='conv5_4')
    return outputs


def tf_get_projections(img, batch_size=batch_size):
    projections = tf.reduce_sum(tf.squeeze(img, -1), axis=1)
    for a in range(1, views):
        angle = -T[a] * (pi / 180)
        rotated_img = tf.contrib.image.rotate(images=img, angles=angle, interpolation='BILINEAR')
        projec = tf.reduce_sum(tf.squeeze(rotated_img, -1), axis=1)
        projections = tf.concat([projections, projec], axis=-1)
    projections = tf.reshape(projections, [batch_size, views, width, 1])

    return projections


def combine_projection(even_pro, odd_pro, batch_size=batch_size):
    full_pro = np.zeros(shape=[batch_size, views, width, 1])
    even_pro = np.expand_dims(np.squeeze(even_pro, 1), -1)
    odd_pro = np.expand_dims(np.squeeze(odd_pro, 1), -1)
    full_pro[:, :, 0::2, :] = even_pro
    full_pro[:, :, 1::2, :] = odd_pro
    return full_pro


if __name__ == '__main__':
    lr = tf.placeholder(dtype=tf.float32, shape=[])  # learning rate
    FD_placeholder = tf.placeholder(dtype=tf.float32, shape=[batch_size, width, width, 1], name='FD_placeholder')
    projection_holder = tf.placeholder(dtype=tf.float32, shape=[batch_size, views, width, 1],
                                         name='projection_holder')

    projection_holder_test = tf.placeholder(dtype=tf.float32, shape=[1, views, width, 1],
                                              name='projection_holder_test_o')
    FD_placeholder_test = tf.placeholder(dtype=tf.float32, shape=[1, width, width, 1], name='FD_placeholder_test')

    with tf.variable_scope('generator_model', reuse=tf.AUTO_REUSE) as scope_generator_model:
        Gz1 = generator(projection_holder, batch_size=batch_size, PI=1)
        Gz_test1 = generator(projection_holder_test, batch_size=1, PI=views)

    with tf.variable_scope('vgg16') as scope:
        vgg_real = vgg_model(FD_placeholder)
        scope.reuse_variables()
        vgg_fake1 = vgg_model(Gz1)

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

    vgg_cost1 = tf.reduce_mean(tf.squared_difference(vgg_real, vgg_fake1))
    mae = tf.reduce_mean(tf.abs(Gz1 - FD_placeholder))
    ssim1 = tf.squeeze(tf.reduce_mean(tf.image.ssim(Gz1, FD_placeholder, 1.0)))
    ssim_loss1 = 1 - ssim1
    g_loss1 = ssim_loss1 * lamda_P + mae + lamda_R * gen_cost1 + lambda_vgg * vgg_cost1

    gen_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_model')
    disc_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_model')
    vgg_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg16')

    total_parameters = 0
    for variable in gen_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)

    # Train the generator
    g_trainer1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss1, var_list=gen_variables)
    # g_trainer2 = tf.train.AdamOptimizer(learning_rate=lr*2).minimize(g_loss2, var_list=gen_variables2)
    with tf.device('/device:GPU:1'):
        d_trainer = tf.train.AdamOptimizer(learning_rate=lr).minimize(disc_loss, var_list=disc_variables)
    #####################################################

    # Patient_IDs = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333']
    Patient_IDs = ['KBCT010140', 'KBCT019104', 'KBCT110038', 'KBCT110041', 'KBCT110087',
                   'KBCT110089', 'KBCT110134', 'KBCT110136', 'KBCT110137',
                   'KBCT110061L', 'KBCT110107R', 'KBCT110136R', 'KBCT110028L', 'KBCT110050R',
                   'KBCT020129L', 'KBCT020102R', 'KBCT020164R', 'KBCT020153L', 'KBCT020076R',
                   'KBCT020005L', 'KBCT010203R', 'KBCT020066R', 'KBCT020077L', 'KBCT020047R',
                   'KBCT010193L', 'KBCT010157R', 'KBCT010198R', 'KBCT010184L', 'KBCT010119L',
                   'KBCT010069L', 'KBCT010154R', 'KBCT010141R']
    Test_Patient_IDs = ['KBCT020021R', 'KBCT020025R', 'KBCT110230L', 'KBCT110149R', 'KBCT110032', 'KBCT019148',
     'KBCT110127L', 'KBCT110053L', 'KBCT110121', 'KBCT010147R']
    #Test_Patient_IDs = ['KBCT110121', 'KBCT010147R']
    #Patient_IDs = Patient_IDs[10:20]
    saver = tf.train.Saver()
    ###############################################################
    dataset_index = 1
    print("enter Session")
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, './model/model.ckpt')
    print("Initialize VGG network ... ")
    weights = np.load('./vgg19.npy', encoding='latin1', allow_pickle=True).item()
    keys = sorted(weights.keys())
    layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
              'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
              'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
              'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']

    for i, k in enumerate(layers):
        print(i, k, weights[k][0].shape, weights[k][1].shape)
        sess.run(vgg_params[2 * i].assign(weights[k][0]))
        sess.run(vgg_params[2 * i + 1].assign(weights[k][1]))
    print('model restored')
    print("Begin training")
    val_lr = LEARNING_RATE
    for iteration in range(start_epoch, num_epoch):
        random.shuffle(Patient_IDs)
        mse_l = []
        ssim_l = []
        w_d_l = []
        vgg_cost_l = []
        gen_l = []
        mse_l2 = []
        ssim_l2 = []
        w_d_l2 = []
        vgg_cost_l2 = []
        gen_l2 = []
        # if iteration % 5 == 0 and iteration != 0:
        val_lr = LEARNING_RATE / np.sqrt(iteration + 1)
        count = 0
        for ID in Patient_IDs:
            break
            filename = "/data/HUIDONG/Koning_o_e_75_views/" + str(ID) + ".h5"
            f = h5py.File(filename, 'r')
            label = np.array(f['image'], dtype=float)
            filtered_sino_e = np.array(f['filtered_sino_e'], dtype=float)
            filtered_sino_o = np.array(f['filtered_sino_o'], dtype=float)
            #fbp_img = np.array(f['fbp_image'], dtype=np.float32)

            f.close()
            label, filtered_sino_e, filtered_sino_o = shuffle(label, filtered_sino_e, filtered_sino_o)
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
                    feed_projection_e = filtered_sino_e[idx[:batch_size]]
                    feed_projection_o = filtered_sino_o[idx[:batch_size]]
                    feed_projection = combine_projection(feed_projection_e, feed_projection_o)
                    _ = sess.run(d_trainer, feed_dict={FD_placeholder: feed_img, projection_holder: feed_projection,
                                                       lr: val_lr})

                batch_label = label[i * batch_size: (i + 1) * batch_size]
                feed_img = np.squeeze(batch_label)
                feed_img = np.expand_dims(feed_img, -1)
                feed_projection_e = filtered_sino_e[i * batch_size: (i + 1) * batch_size]
                feed_projection_o = filtered_sino_o[i * batch_size: (i + 1) * batch_size]
                feed_projection = combine_projection(feed_projection_e, feed_projection_o)

                _, _mae, _ssim, _gen_cost, _difference, _vgg_cost = sess.run(
                    [g_trainer1, mae, ssim1, gen_cost1, difference1, vgg_cost1],
                    feed_dict={FD_placeholder: feed_img,
                               projection_holder: feed_projection,
                               lr: val_lr})

                print("Epoch:{} {} {} mae:{} ssim:{} gen_cost:{} differencee:{} vgg_cost:{}".format(iteration, ID, i,
                                                                                                    _mae, _ssim,
                                                                                                    _gen_cost,
                                                                                                    _difference,
                                                                                                    _vgg_cost))

                mse_l.append(_mae)
                ssim_l.append(_ssim)
                vgg_cost_l.append(_vgg_cost)
                w_d_l.append(_difference)
                gen_l.append(_gen_cost)

        print("writing file")
        with open("50_mae.txt", 'a') as f:
            for a in mse_l:
                f.write('{}\n'.format(a))

        with open("50_ssim.txt", 'a') as f:
            for a in ssim_l:
                f.write('{}\n'.format(a))

        with open("50_w_d.txt", 'a') as f:
            for a in w_d_l:
                f.write('{}\n'.format(a))

        with open("50_gen_c.txt", 'a') as f:
            for a in gen_l:
                f.write('{}\n'.format(a))

        with open("50_vgg.txt", 'a') as f:
            for a in vgg_cost_l:
                f.write('{}\n'.format(a))
        saver.save(sess, './model/model.ckpt')


        print("testing dataset")
        for ID in Test_Patient_IDs:
            print("Testing Patient:{}".format(ID))
            dir = '/home/xiehuidong/DEER/Koning_75_/' + str(ID)
            os.mkdir(dir)
            dir = '/home/xiehuidong/DEER/Koning_75_/' + str(ID) + '/real/'
            os.mkdir(dir)
            dir = '/home/xiehuidong/DEER/Koning_75_/' + str(ID) + '/result/'
            os.mkdir(dir)
            dir = '/home/xiehuidong/DEER/Koning_75_/' + str(ID) + '/fbp/'
            os.mkdir(dir)
            #exit(0)
            filename = "/data/HUIDONG/Koning_o_e_75_views/" + str(ID) + ".h5"
            c = 0
            f_t = h5py.File(filename, 'r')
            label_t = np.array(f_t['image'], dtype=float)
            label_s_e = np.array(f_t['filtered_sino_e'], dtype=np.float32)
            label_s_o = np.array(f_t['filtered_sino_o'], dtype=np.float32)
            label_fbp_img = np.array(f_t['fbp_images'], dtype=np.float32)
            f_t.close()
            test_mse = []
            test_ssim = []
            test_mse2 = []
            test_ssim2 = []
            test_mse_p = []
            test_mse_p2 = []
            testing_results = []

            for c in range(len(label_s_o) // 1):
                print(c)
                test_img = np.squeeze(label_t[c * 1: (c + 1) * 1])
                test_img = np.expand_dims(test_img, 0)
                test_img = np.expand_dims(test_img, -1)
                test_sinogram_o = label_s_o[c * 1: (c + 1) * 1]
                test_sinogram_e = label_s_e[c * 1: (c + 1) * 1]
                test_sinogram = combine_projection(test_sinogram_e, test_sinogram_o, 1)

                with tf.variable_scope('generator_model') as scope:
                    scope.reuse_variables()
                    estimated = sess.run(Gz_test1, feed_dict={FD_placeholder_test: test_img,
                                                              projection_holder_test: test_sinogram})
                S = []
                P = []
                MSE = []
                e_1 = normalize(np.squeeze(estimated[0]))
                testing_results.append(e_1)

            testing_results = np.asarray(testing_results)
            img_path = os.path.join(dataset_path, ID)
            filenames = os.listdir(img_path)
            filenames = sorted(filenames)
            img_slices = []
            for i in filenames:
                if i[0] != '.' and ('.IMA' in i or '.dcm' in i):
                    img_slices.append(pydicom.read_file(os.path.join(img_path, i)))
            img_pixel = [x.pixel_array for x in img_slices]
            img_pixel = np.array(img_pixel)
            for index in range(testing_results.shape[0]):
                print("fbp::{}".format(index))
                #if np.all(img_pixel[index]):
                #    continue
                fn = './' + str(ID) + '/result/' + str(index) + '.dcm'
                write_dicom(img_slices[index], testing_results[index], fn)
                fn = './' + str(ID) + '/real/' + str(index) + '.dcm'
                real_img = normalize(np.squeeze(label_t[index]))
                write_dicom(img_slices[index], real_img, fn)
                #fbp_img = radon(np.squeeze(label_t[index]), theta, circle=True)
                #fbp_img = iradon(fbp_img, theta, circle=True)
                fbp_img = normalize(np.squeeze(label_fbp_img[index]))
                fn = './' + str(ID) + '/fbp/' + str(index) + '.dcm'
                write_dicom(img_slices[index], fbp_img, fn)
        exit(0)
