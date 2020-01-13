"""
find center using focal loss.

fail.

"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tools.DataHandler_racoon_test1 as DataHandler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load data.

dm = DataHandler.DataHandler(oval_ratio=0.02)

input = tf.placeholder(np.float32, [None, None, None, 3])
label_center_1_by_1 = tf.placeholder(np.float32, [None, None, None, 1])
# label_center_2_by_2 = tf.placeholder(np.float32, [None, None, None, 1])
# label_center_4_by_4 = tf.placeholder(np.float32, [None, None, None, 1])
# label_center_8_by_8 = tf.placeholder(np.float32, [None, None, None, 1])
training = tf.placeholder(np.bool)


########## LAYER BOX #############
def conv_block(input, traget_dim, pooling, training):
    filter = tf.Variable(tf.random_normal(shape=[3, 3, input.get_shape().as_list()[-1], traget_dim], stddev=0.1))
    after_conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
    after_acti = tf.nn.relu(after_conv)
    final = tf.layers.batch_normalization(after_acti, center=True, scale=True, training=training)

    if pooling:
        final = tf.nn.max_pool(final, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    print(final.shape)

    return final


def conv_block_final(input, traget_dim, training):
    filter = tf.Variable(tf.random_normal(shape=[3, 3, input.get_shape().as_list()[-1], traget_dim], stddev=0.1))
    after_conv = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
    final = tf.layers.batch_normalization(after_conv, center=True, scale=True, training=training)

    print(final.shape)

    return final


def deconv_block(input, target_dim, restore_shape, training):
    filter = tf.Variable(
        tf.random_normal(shape=[3, 3, target_dim, input.get_shape().as_list()[3]], stddev=0.1))
    after_conv = tf.nn.conv2d_transpose(input,
                                        output_shape=[restore_shape[0], restore_shape[1],
                                                      restore_shape[2], target_dim],
                                        filter=filter, strides=[1, 2, 2, 1], padding="SAME")

    after_acti = tf.nn.relu(after_conv)
    after_batch = tf.layers.batch_normalization(after_acti, center=True, scale=True, training=training)

    print(after_batch.shape)

    return after_batch


##################################


model_4 = conv_block(input, 32, False, training)  # 1
model = conv_block(model_4, 32, True, training)  # /2
model_3 = conv_block(model, 64, False, training)  # /2
model = conv_block(model_3, 64, True, training)  # /4
model_2 = conv_block(model, 128, False, training)  # /4
model = conv_block(model_2, 128, True, training)  # /8
model_1 = conv_block(model, 256, False, training)  # /8
model = conv_block(model_1, 256, True, training)  # /16
model = conv_block(model, 512, False, training)  # /16
model = conv_block(model, 512, False, training)  # /16

########## center branch ###########
model_center = deconv_block(model, 256, tf.shape(model_1), training)

model_center = model_center + conv_block(model_1, 256, False, training)
# model_8_by_8 = conv_block(model_center, 1, False, training)

model_center = conv_block(model_center, 256, False, training)
model_center = conv_block(model_center, 256, False, training)
model_center = deconv_block(model_center, 128, tf.shape(model_2), training)

model_center = model_center + conv_block(model_2, 128, False, training)
# model_4_by_4 = conv_block(model_center, 1, False, training)

model_center = conv_block(model_center, 128, False, training)
model_center = conv_block(model_center, 128, False, training)
model_center = deconv_block(model_center, 64, tf.shape(model_3), training)

model_center = model_center + conv_block(model_3, 64, False, training)
# model_2_by_2 = conv_block(model_center, 1, False, training)

model_center = conv_block(model_center, 64, False, training)
model_center = conv_block(model_center, 64, False, training)
model_center = deconv_block(model_center, 32, tf.shape(model_4), training)

model_center = model_center + conv_block(model_4, 32, False, training)

model_center = conv_block(model_center, 32, False, training)
model_center = conv_block(model_center, 32, False, training)
model_1_by_1_prev_reshape = conv_block_final(model_center, 1, training)
model_1_by_1_prev_softmax = tf.reshape(model_1_by_1_prev_reshape, [-1])
output_1_by_1 = tf.nn.softmax(model_1_by_1_prev_softmax)

label_1_by_1 = tf.reshape(label_center_1_by_1, [-1])


# label_2_by_2 = tf.reshape(label_center_2_by_2, [-1])
# input_2_by_2 = tf.reshape(model_2_by_2, [-1])
# label_4_by_4 = tf.reshape(label_center_4_by_4, [-1])
# input_4_by_4 = tf.reshape(model_4_by_4, [-1])
# label_8_by_8 = tf.reshape(label_center_8_by_8, [-1])
# input_8_by_8 = tf.reshape(model_8_by_8, [-1])


# def get_focal_loss(input, label):
#     # label_count = tf.constant(0)
#     # loss_sum = tf.constant(0.)
#     #
#     # def cond(i, loss_sum):
#     #     print(tf.size(input))
#     #     return tf.less(i, tf.size(input))
#     #
#     # def body(i, loss_sum):
#     #     print("body_start")
#     #
#     #     tensor_1 = tf.squeeze(tf.gather(input, i))
#     #
#     #     elements = input
#     #
#     #     def get_loss_each_element(element):
#     #         loss = tf.cond(tf.equal(tensor_1, tf.constant(1.)),
#     #                        lambda: tf.square(tf.constant(1.) - element) * tf.log(element),
#     #                        lambda: tf.pow(tf.constant(1.) - tensor_1, tf.constant(4.)) * tf.square(
#     #                            tf.log(element)) * tf.log(tf.constant(1.) - element))
#     #
#     #         return loss
#     #
#     #     loss_list = tf.map_fn(get_loss_each_element, elements)
#     #
#     #     loss_sum = tf.add(loss_sum, tf.reduce_sum(loss_list))
#     #
#     #     return tf.add(i, 1), loss_sum
#     #
#     # result = tf.while_loop(cond, body, [label_count, loss_sum])
#     # x is when true, y is when false.
#     loss_list = tf.where(condition=tf.cast(label, tf.bool),
#                          x=tf.square(tf.constant(1.) - input) * tf.log(input),
#                          y=tf.square(tf.log(input)) * tf.log(tf.constant(1.) - input))
#
#     result = tf.reduce_sum(loss_list)
#
#     nums_1 = tf.cast(tf.count_nonzero(label), tf.float32)
#     loss_sum = (-1. * tf.reciprocal(nums_1)) * result
#
#     return loss_sum
def get_focal_loss_v2(input, label):
    loss_list = tf.where(condition=tf.cast(label, tf.bool),
                         x=tf.square(tf.constant(1.) - input) * tf.log(input),
                         y=tf.square(input) * tf.log(tf.constant(1.) - input))

    result = tf.reduce_sum(loss_list)

    nums_1 = tf.cast(tf.count_nonzero(label), tf.float32)
    loss_sum = (-1. * tf.reciprocal(nums_1)) * result

    return loss_sum


get_focal_loss_1by1 = get_focal_loss_v2(output_1_by_1, label_1_by_1)
# get_focal_loss_2by2 = get_focal_loss(input_2_by_2, label_2_by_2)
# get_focal_loss_4by4 = get_focal_loss(input_4_by_4, label_4_by_4)
# get_focal_loss_8by8 = get_focal_loss(input_8_by_8, label_8_by_8)

# loss_center_1_by_1 = tf.sqrt(tf.reduce_mean(tf.square((label_center_1_by_1) - (model_1_by_1))))
# loss_center_2_by_2 = tf.sqrt(tf.reduce_mean(tf.square((label_center_2_by_2) - (model_2_by_2))))
# loss_center_4_by_4 = tf.sqrt(tf.reduce_mean(tf.square((label_center_4_by_4) - (model_4_by_4))))
# loss_center_8_by_8 = tf.sqrt(tf.reduce_mean(tf.square((label_center_8_by_8) - (model_8_by_8))))

# total_loss = loss_center_1_by_1 + loss_center_2_by_2 + loss_center_4_by_4 + loss_center_8_by_8

# total_loss = get_focal_loss_1by1 + get_focal_loss_2by2 + get_focal_loss_4by4 + get_focal_loss_8by8
total_loss = get_focal_loss_1by1
optimizer_center = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(18000):
        batch_x, train_images1by1, _ = dm.nextBatch(1)

        train_images1by1 = np.expand_dims(train_images1by1, axis=-1)

        # ## test
        # print("loss model: {}".format(sess.run(total_loss, feed_dict={input: batch_x,
        #                                                               label_center_1_by_1: train_images1by1,
        #                                                               training: False})))

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        _ = sess.run([optimizer_center, extra_update_ops],
                     feed_dict={input: batch_x,
                                label_center_1_by_1: train_images1by1,
                                training: True})

        output = sess.run(output_1_by_1, feed_dict={input: batch_x,
                                                    label_center_1_by_1: train_images1by1,
                                                    training: False})

        label = np.squeeze(train_images1by1)

        # dinstance = np.sqrt(np.square(np.where(label == np.max(label))[0] -
        #                               np.where(np.reshape(output, np.shape(label)) == np.max(
        #                                   np.reshape(output, np.shape(label))))[0])
        #                     + np.square(np.where(label == np.max(label))[1] -
        #                                 np.where(np.reshape(output, np.shape(label)) == np.max(
        #                                     np.reshape(output, np.shape(label))))[
        #                                     1]))

        print(i, "th", "loss model: {}".format(sess.run(total_loss, feed_dict={input: batch_x,
                                                                               label_center_1_by_1: train_images1by1,
                                                                               training: False})))

# , "label", np.where(label == np.max(label)), "output", np.where(
#     np.reshape(output, np.shape(label)) == np.max(np.reshape(output, np.shape(label))))


        # plt.imshow(np.reshape(output, np.shape(label))*255)
        # plt.show()
        # plt.imshow(label*255)
        # plt.show()
