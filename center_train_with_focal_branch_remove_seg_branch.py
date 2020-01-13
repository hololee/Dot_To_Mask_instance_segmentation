import tensorflow as tf
import numpy as np
import os
import DataManager
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# load data.

dm = DataManager.DataManager()

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


##################################

model = conv_block(input, 32, False, training)
model = conv_block(model, 32, True, training)
model = conv_block(model, 64, False, training)
model = conv_block(model, 64, True, training)
model = conv_block(model, 128, False, training)
model = conv_block(model, 128, True, training)
model = conv_block(model, 256, False, training)
model = conv_block(model, 256, True, training)
model = conv_block(model, 512, False, training)
model = conv_block(model, 512, False, training)

##########center seg branch ###########
# model_center = deconv_block(model, 256, training)
# model_center = conv_block(model_center, 256, False, training)
# model_center = conv_block(model_center, 256, False, training)
# model_center = deconv_block(model_center, 128, training)
# model_center = conv_block(model_center, 128, False, training)
# model_center = conv_block(model_center, 128, False, training)
# model_center = deconv_block(model_center, 64, training)
# model_center = conv_block(model_center, 64, False, training)
# model_center = conv_block(model_center, 64, False, training)
# model_center = deconv_block(model_center, 32, training)
# model_center = conv_block(model_center, 32, False, training)
# model_center = conv_block(model_center, 32, False, training)
# out_center = final_block(model_center, 1)
# out_center = tf.nn.sigmoid(out_center)

########## focal branch ###########
model_focal = deconv_block(model, 256, training)
model_focal = conv_block(model_focal, 256, False, training)
model_focal = conv_block(model_focal, 256, False, training)
model_focal = deconv_block(model_focal, 128, training)
model_focal = conv_block(model_focal, 128, False, training)
model_focal = conv_block(model_focal, 128, False, training)
model_focal = deconv_block(model_focal, 64, training)
model_focal = conv_block(model_focal, 64, False, training)
model_focal = conv_block(model_focal, 64, False, training)
model_focal = deconv_block(model_focal, 32, training)
model_focal = conv_block(model_focal, 32, False, training)
model_focal = conv_block(model_focal, 32, False, training)
out_focal_before_softmax = final_block(model_focal, 1)
out_focal_before_softmax = tf.reshape(out_focal_before_softmax, [-1])
out_focal = tf.nn.sigmoid(out_focal_before_softmax)

# loss_center = tf.sqrt(tf.reduce_mean(tf.square(label_center - out_center)))
loss_focal = get_focal_loss(out_focal, tf.reshape(label_center, [-1]))
# loss_segmentation = tf.sqrt(tf.reduce_mean(tf.square(label_segmentation - tf.squeeze(out_segmentation))))

total_loss = loss_focal

optimizer_t = tf.train.AdamOptimizer(learning_rate=0.001).minimize(total_loss)

# optimizer_segmentation = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_segmentation)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(18000):
        batch_x, batch_center, _ = dm.next_batch(dm.data_x, dm.data_center, dm.data_segmentation, 1)
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
                sess.run(tf.reshape(out_focal, shape=[512, 512]), feed_dict={input: batch_x, training: False})))

            ax3 = fig.add_subplot(2, 1, 2)
            ax3.title.set_text("label")
            ax3.imshow(np.squeeze(batch_center))

            # ax2 = fig.add_subplot(2, 2, 4)
            # ax2.title.set_text("sum")
            # ax2.imshow(np.squeeze(sess.run(out_center, feed_dict={input: batch_x, training: False})) + np.squeeze(
            #     sess.run(tf.reshape(out_focal, shape=[512, 512]), feed_dict={input: batch_x, training: False})))

            plt.show()

        if i == 12000:
            print("stop")
