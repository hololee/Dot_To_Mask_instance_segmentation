import tensorflow as tf
import numpy as np
import os
import DataManagerCenterTrain
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# load data.

dm = DataManagerCenterTrain.DataManager()

input = tf.placeholder(np.float32, [None, 512, 512, 3])
label_center = tf.placeholder(np.float32, [None, 512, 512, 1])
training = tf.placeholder(np.bool)


########## LAYER BOX #############
def conv_block(input, traget_dim, pooling, training):
    filter = tf.Variable(tf.random_normal(shape=[3, 3, input.get_shape().as_list()[-1], traget_dim], stddev=0.1))
    after_conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
    after_acti = tf.nn.relu(after_conv, "22d")
    final = tf.layers.batch_normalization(after_acti, center=True, scale=True, training=training)

    if pooling:
        final = tf.nn.max_pool(final, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    print(final.shape)

    return final


def final_block(input_f, traget_dim):
    filter = tf.Variable(tf.random_normal(shape=[3, 3, input_f.get_shape().as_list()[-1], traget_dim], stddev=0.1))
    after_conv = tf.nn.conv2d(input_f, filter, strides=[1, 1, 1, 1], padding="SAME")
    final = tf.layers.batch_normalization(after_conv, center=True, scale=True, training=training)

    return final


def deconv_block(input_de, target_dim, training):
    filter = tf.Variable(
        tf.random_normal(shape=[3, 3, target_dim, input_de.get_shape().as_list()[3]], stddev=0.1))
    after_conv = tf.nn.conv2d_transpose(input_de,
                                        output_shape=[1, input_de.get_shape().as_list()[1] * 2,
                                                      input_de.get_shape().as_list()[2] * 2, target_dim],
                                        filter=filter, strides=[1, 2, 2, 1], padding="SAME")

    after_acti = tf.nn.relu(after_conv, "5d")
    after_batch = tf.layers.batch_normalization(after_acti, center=True, scale=True, training=training)

    print(after_batch.shape)

    return after_batch


def get_focal_loss(out_of_model, label):
    loss_list = tf.where(condition=tf.cast(label, tf.bool),
                         x=tf.square(tf.constant(1.) - out_of_model) * tf.log(out_of_model + 1e-17),
                         y=tf.square(out_of_model) * tf.log(tf.constant(1.) - out_of_model + 1e-17))

    result = tf.reduce_sum(loss_list)

    nums_1 = tf.cast(tf.count_nonzero(label), tf.float32)
    loss_sum = (-1. * tf.reciprocal(nums_1)) * result

    return loss_sum


def hourglassModule(input, finish=False):
    ##################################
    # input = 256,256, 32
    model_b = conv_block(input, 64, False, training)  # 256
    model_bb = conv_block(model_b, 64, True, training)  # 128
    model_c = conv_block(model_bb, 128, False, training)  # 128
    model_cc = conv_block(model_c, 128, True, training)  # 64
    model_d = conv_block(model_cc, 256, False, training)  # 64
    model_dd = conv_block(model_d, 256, True, training)  # 32
    model = conv_block(model_dd, 512, False, training)  # 32
    # model = conv_block(model, 512, False, training)  # 32

    ########## focal branch ###########
    model_focal = deconv_block(model, 256, training) + conv_block(model_d, 256, False, training)  # 64,64,256
    model_focal = conv_block(model_focal, 256, False, training)
    # model_focal = conv_block(model_focal, 256, False, training)
    model_focal = deconv_block(model_d, 128, training) + conv_block(model_c, 128, False, training)
    model_focal = conv_block(model_focal, 128, False, training)
    # model_focal = conv_block(model_focal, 128, False, training)
    model_focal = deconv_block(model_focal, 64, training) + conv_block(model_b, 64, False, training)
    model_focal = conv_block(model_focal, 32, False, training)
    # model_focal = conv_block(model_focal, 64, False, training)
    # model_focal = deconv_block(model_focal, 32, training)

    if finish:
        model_focal = deconv_block(model_focal, 32, training)
        model_focal = conv_block(model_focal, 32, False, training)
        # model_focal = conv_block(model_focal, 32, False, training)
        model_focal = final_block(model_focal, 1)
        model_focal = tf.reshape(model_focal, [-1])
        model_focal = tf.nn.sigmoid(model_focal)

    return model_focal


model_a = conv_block(input, 32, False, training)  # 512, 512 , 32
model_aa = conv_block(model_a, 32, True, training)  # 256

out_focal = hourglassModule(model_aa)
out_focal = hourglassModule(out_focal)
out_focal = hourglassModule(out_focal, True)

# loss_center = tf.sqrt(tf.reduce_mean(tf.square(label_center - out_center)))
loss_focal = get_focal_loss(out_focal, tf.reshape(label_center, [-1]))
# loss_segmentation = tf.sqrt(tf.reduce_mean(tf.square(label_segmentation - tf.squeeze(out_segmentation))))

total_loss = loss_focal

optimizer_t = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)

# optimizer_segmentation = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_segmentation)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(18000):
        batch_x, batch_center, _ = dm.next_batch(dm.train_data_x, dm.train_data_center, dm.train_data_segmentation, 1)
        batch_center = np.expand_dims(batch_center, axis=-1)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        sess.run([optimizer_t, extra_update_ops],
                 feed_dict={input: batch_x, label_center: batch_center, training: True})

        # print("loss center: {}".format(sess.run(loss_center,
        #                                         feed_dict={input: batch_x, label_center: batch_center,
        #                                                    training: False})))
        print("loss focal: {}".format(sess.run(loss_focal,
                                               feed_dict={input: batch_x, label_center: batch_center,
                                                          training: False})))

        if (i % 200) == 0:
            fig = plt.figure()

            # ax1 = fig.add_subplot(2, 1, 1)
            # ax1.title.set_text("center")
            # ax1.imshow(np.squeeze(sess.run(out_center, feed_dict={input: batch_x, training: False})))

            ax2 = fig.add_subplot(2, 1, 1)
            ax2.title.set_text("focal")
            ax2.imshow(np.squeeze(
                sess.run(tf.reshape(out_focal, shape=[512, 512]),
                         feed_dict={input: np.expand_dims(dm.test_data_x[0], axis=0), training: False})))

            ax3 = fig.add_subplot(2, 1, 2)
            ax3.title.set_text("label")
            ax3.imshow(np.squeeze(np.expand_dims(dm.test_data_center[0], axis=0)))
            # ax2 = fig.add_subplot(2, 2, 4)
            # ax2.title.set_text("sum")
            # ax2.imshow(np.squeeze(sess.run(out_center, feed_dict={input: batch_x, training: False})) + np.squeeze(
            #     sess.run(tf.reshape(out_focal, shape=[512, 512]), feed_dict={input: batch_x, training: False})))

            plt.show()

        if i == 12000:
            print("stop")
