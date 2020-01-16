import tensorflow as tf
import numpy as np
import os
import DataManagerCenterTrain
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# load data.

dm = DataManagerCenterTrain.DataManager()

input = tf.placeholder(np.float32, [None, 512, 512, 3], name="input")
label_match = tf.placeholder(np.float32, [None, 512, 512, 1], name="label_match")  # one leaf.
label_segmentation = tf.placeholder(np.float32, [None, 512, 512, 1], name="label_segmentation")  # total leaves.
individual_points = tf.placeholder(np.float32, [None, 512, 512, 1], name="individual_points")  # one center_point.
label_center = tf.placeholder(np.float32, [None, 512, 512, 1], name="label_center")  # total center_points.
training = tf.placeholder(np.bool, name="training")


# --------------  params  ----------------


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

model_32 = conv_block(input, 32, False, training)
model_32_2 = conv_block(model_32, 32, True, training)
model_64 = conv_block(model_32_2, 64, False, training)
model_64_2 = conv_block(model_64, 64, True, training)
model_128 = conv_block(model_64_2, 128, False, training)
model_128_2 = conv_block(model_128, 128, True, training)
model_256 = conv_block(model_128_2, 256, False, training)
model_256_2 = conv_block(model_256, 256, True, training)
model_512 = conv_block(model_256_2, 512, False, training)
model_512_2 = conv_block(model_512, 512, False, training)

########## focal branch ###########
model_focal = deconv_block(model_512_2, 256, training)
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

########## match branch ###########
model_segmentation = deconv_block(model_512_2, 256, training)
model_segmentation = tf.concat([model_segmentation, tf.image.resize_images(individual_points, (
    model_segmentation.get_shape().as_list()[1], model_segmentation.get_shape().as_list()[2]))], axis=-1)
model_segmentation = conv_block(model_segmentation, 256, False, training)
model_segmentation = conv_block(model_segmentation, 256, False, training)
model_segmentation = deconv_block(model_segmentation, 128, training)
model_segmentation = tf.concat([model_segmentation, tf.image.resize_images(individual_points, (
    model_segmentation.get_shape().as_list()[1], model_segmentation.get_shape().as_list()[2]))], axis=-1)
model_segmentation = conv_block(model_segmentation, 128, False, training)
model_segmentation = conv_block(model_segmentation, 128, False, training)
model_segmentation = deconv_block(model_segmentation, 64, training)
model_segmentation = tf.concat([model_segmentation, tf.image.resize_images(individual_points, (
    model_segmentation.get_shape().as_list()[1], model_segmentation.get_shape().as_list()[2]))], axis=-1)
model_segmentation = conv_block(model_segmentation, 64, False, training)
model_segmentation = conv_block(model_segmentation, 64, False, training)
model_segmentation = deconv_block(model_segmentation, 32, training)
model_segmentation = tf.concat([model_segmentation, tf.image.resize_images(individual_points, (
    model_segmentation.get_shape().as_list()[1], model_segmentation.get_shape().as_list()[2]))], axis=-1)
model_segmentation = conv_block(model_segmentation, 32, False, training)
out_match = conv_block(model_segmentation, 1, False, training)
out_match = tf.nn.sigmoid(out_match)

loss_focal = get_focal_loss(out_focal, tf.reshape(label_center, [-1]))
# loss_segmentation = tf.sqrt(tf.reduce_mean(tf.square(label_segmentation - tf.squeeze(out_segmentation))))

loss_match = tf.sqrt(tf.reduce_mean(tf.pow(label_match - out_match, 4)))

alpha = 1.
# beta = 1.
gamma = 1.

optimizer_step1 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_focal)
optimizer_step2 = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_match)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_data = len(dm.data_x)
    batch_size = 1

    iterations = total_data // batch_size

    total_epoch = 3000

    for epoch in range(total_epoch):

        print("==================== EPOCH {} ====================".format(epoch))

        for iteration in range(iterations):
            origin_image, center_image, segmentation_image, one_center, one_segmentation = dm.next_batch_with_each_center(
                1)
            center_image = np.expand_dims(center_image, axis=-1)
            segmentation_image = np.expand_dims(segmentation_image, axis=-1)

            one_center = np.squeeze(one_center)
            one_segmentation = np.squeeze(one_segmentation)

            # step
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # for sub_iteration in range(len(one_center)):

            # sub_iteration = int(np.random.choice(range(len(one_center)), 1))
            for sub_iteration in range(len(one_center)):
                one_center_image = one_center[sub_iteration]
                one_center_image = np.expand_dims(one_center_image, axis=0)
                one_center_image = np.expand_dims(one_center_image, axis=-1)
                one_segmentation_image = one_segmentation[sub_iteration]
                one_segmentation_image = np.expand_dims(one_segmentation_image, axis=0)
                one_segmentation_image = np.expand_dims(one_segmentation_image, axis=-1)

                sess.run([optimizer_step1, extra_update_ops],
                         feed_dict={input: origin_image,
                                    label_center: center_image,
                                    label_segmentation: segmentation_image,
                                    individual_points: one_center_image,
                                    label_match: one_segmentation_image,
                                    training: True})

                sess.run([optimizer_step2, extra_update_ops],
                         feed_dict={input: origin_image,
                                    label_center: center_image,
                                    label_segmentation: segmentation_image,
                                    individual_points: one_center_image,
                                    label_match: one_segmentation_image,
                                    training: True})

                print("iteration{}:{}   __ loss_match: {}  __ loss_focal: {}".format(iteration,
                                                                                     sub_iteration,

                                                                                     sess.run(
                                                                                         loss_match,
                                                                                         feed_dict={
                                                                                             input: origin_image,
                                                                                             label_center: center_image,
                                                                                             label_segmentation: segmentation_image,
                                                                                             individual_points: one_center_image,
                                                                                             label_match: one_segmentation_image,
                                                                                             training: False}),
                                                                                     sess.run(
                                                                                         loss_focal,
                                                                                         feed_dict={
                                                                                             input: origin_image,
                                                                                             label_center: center_image,
                                                                                             label_segmentation: segmentation_image,
                                                                                             individual_points: one_center_image,
                                                                                             label_match: one_segmentation_image,
                                                                                             training: False})))

                # if (iteration % 5) == 0:
                #     fig = plt.figure()
                #
                #     ax1 = fig.add_subplot(1, 3, 1)
                #     ax1.title.set_text("out_match")
                #     ax1.imshow(np.squeeze(sess.run(tf.reshape(out_match, shape=[512, 512]), feed_dict={input: origin_image,
                #                                                                                        label_center: center_image,
                #                                                                                        label_segmentation: segmentation_image,
                #                                                                                        individual_points: one_center_image,
                #                                                                                        label_match: one_segmentation_image,
                #                                                                                        training: False})))
                #
                #     ax2 = fig.add_subplot(1, 3, 2)
                #     ax2.title.set_text("out_focal")
                #     ax2.imshow(np.squeeze(
                #         sess.run(tf.reshape(out_focal, shape=[512, 512]),
                #                  feed_dict={input: origin_image,
                #                             label_center: center_image,
                #                             label_segmentation: segmentation_image,
                #                             individual_points: one_center_image,
                #                             label_match: one_segmentation_image,
                #                             training: False})))
                #
                #     ax3 = fig.add_subplot(1, 3, 3)
                #     ax3.title.set_text("out_segmentation")
                #     ax3.imshow(np.squeeze(
                #         sess.run(out_segmentation,
                #                  feed_dict={input: origin_image,
                #                             label_center: center_image,
                #                             label_segmentation: segmentation_image,
                #                             individual_points: one_center_image,
                #                             label_match: one_segmentation_image,
                #                             training: False})))
                #
                #     # ax2 = fig.add_subplot(2, 2, 4)
                #     # ax2.title.set_text("sum")
                #     # ax2.imshow(np.squeeze(sess.run(out_center, feed_dict={input: batch_x, training: False})) + np.squeeze(
                #     #     sess.run(tf.reshape(out_focal, shape=[512, 512]), feed_dict={input: batch_x, training: False})))
                #
                #     plt.show()

        if epoch == np.max(range(total_epoch)):
            print("stop")
