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
label_center = tf.placeholder(np.float32, [None, 512, 512])
label_segmentation = tf.placeholder(np.float32, [None, 512, 512])
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


def deconv_block(input, target_dim, training):
    filter = tf.Variable(
        tf.random_normal(shape=[3, 3, target_dim, input.get_shape().as_list()[3]], stddev=0.1))
    after_conv = tf.nn.conv2d_transpose(input,
                                        output_shape=[1, input.get_shape().as_list()[1] * 2,
                                                      input.get_shape().as_list()[2] * 2, target_dim],
                                        filter=filter, strides=[1, 2, 2, 1], padding="SAME")

    after_acti = tf.nn.relu(after_conv, "5d")
    after_batch = tf.layers.batch_normalization(after_acti, center=True, scale=True, training=training)

    print(after_batch.shape)

    return after_batch


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

##########center branch ###########
model_center = deconv_block(model, 256, training)
model_center = conv_block(model_center, 256, False, training)
model_center = conv_block(model_center, 256, False, training)
model_center = deconv_block(model_center, 128, training)
model_center = conv_block(model_center, 128, False, training)
model_center = conv_block(model_center, 128, False, training)
model_center = deconv_block(model_center, 64, training)
model_center = conv_block(model_center, 64, False, training)
model_center = conv_block(model_center, 64, False, training)
model_center = deconv_block(model_center, 32, training)
model_center = conv_block(model_center, 32, False, training)
model_center = conv_block(model_center, 32, False, training)
out_center = conv_block(model_center, 1, False, training)

########## segmentation branch ###########
model_segmentation = deconv_block(model, 256, training)
model_segmentation = conv_block(model_segmentation, 256, False, training)
model_segmentation = conv_block(model_segmentation, 256, False, training)
model_segmentation = deconv_block(model_segmentation, 128, training)
model_segmentation = conv_block(model_segmentation, 128, False, training)
model_segmentation = conv_block(model_segmentation, 128, False, training)
model_segmentation = deconv_block(model_segmentation, 64, training)
model_segmentation = conv_block(model_segmentation, 64, False, training)
model_segmentation = conv_block(model_segmentation, 64, False, training)
model_segmentation = deconv_block(model_segmentation, 32, training)
model_segmentation = conv_block(model_segmentation, 32, False, training)
model_segmentation = conv_block(model_segmentation, 32, False, training)
out_segmentation = conv_block(model_segmentation, 1, False, training)

loss_center = tf.sqrt(tf.reduce_mean(tf.square(label_center - tf.squeeze(out_center))))
loss_segmentation = tf.sqrt(tf.reduce_mean(tf.square(label_segmentation - tf.squeeze(out_segmentation))))

optimizer_center = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_center)
optimizer_segmentation = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_segmentation)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(18000):
        batch_x, batch_center, batch_segmentation = dm.next_batch(dm.data_x, dm.data_center, dm.data_segmentation, 1)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        _ = sess.run([optimizer_center, optimizer_segmentation, extra_update_ops],
                     feed_dict={input: batch_x, label_center: batch_center, label_segmentation: batch_segmentation,
                                training: True})

        print("loss center: {}".format(sess.run(loss_center,
                                                feed_dict={input: batch_x, label_center: batch_center,
                                                           training: False})))
        print("loss segmentation: {}".format(sess.run(loss_segmentation,
                                                      feed_dict={input: batch_x, label_segmentation: batch_segmentation,
                                                                 training: False})))

        if (i % 200) == 0:
            plt.subplot(1, 3, 1)
            plt.imshow(np.squeeze(sess.run(out_center, feed_dict={input: batch_x, training: False})))

            plt.subplot(1, 3, 2)
            plt.imshow(np.squeeze(sess.run(out_segmentation, feed_dict={input: batch_x, training: False})))

            plt.subplot(1, 3, 3)
            plt.imshow(np.squeeze(batch_x))

            plt.show()
