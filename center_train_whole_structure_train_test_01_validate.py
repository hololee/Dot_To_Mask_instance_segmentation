import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
import DataManagerCenterTrain
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

grayscale_list = np.linspace(0.5, 1, 18)

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

######## summation branch #########
s_model_32 = conv_block(individual_points, 32, False, training)
s_model_32_2 = conv_block(s_model_32, 32, True, training)
s_model_64 = conv_block(s_model_32_2, 64, False, training)
s_model_64_2 = conv_block(s_model_64, 64, True, training)
s_model_128 = conv_block(s_model_64_2, 128, False, training)
s_model_128_2 = conv_block(s_model_128, 128, True, training)
s_model_256 = conv_block(s_model_128_2, 256, False, training)
s_model_256_2 = conv_block(s_model_256, 256, True, training)
s_model_512 = conv_block(s_model_256_2, 512, False, training)
s_model_512_2 = conv_block(s_model_512, 512, False, training)

########## match branch ###########
model_segmentation = deconv_block(s_model_512_2 + model_512_2, 256, training)
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
out_match = conv_block(model_segmentation, 1, False, training)
out_match = tf.nn.sigmoid(out_match)

loss_focal = get_focal_loss(out_focal, tf.reshape(label_center, [-1]))
loss_match = tf.sqrt(tf.reduce_mean(tf.pow(label_match - out_match, 4)))

optimizer_step1 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_focal)
optimizer_step2 = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_match)

with tf.Session() as sess:
    # load data.
    saver = tf.train.Saver()
    saver.restore(sess, "/data1/LJH/Dot_To_Mask_instance_segmentation/saved_models/test_01/model_epoch_130")

    IOU_LIST = []

    # NOTICE : for images.
    for origin_index in range(len(dm.get_test_data())):
        origin_image, center_image, segmentation_image, one_center, one_segmentation, color_label = dm.get_test_data()

        origin_image = origin_image[origin_index]
        origin_image = np.expand_dims(origin_image, axis=0)

        center_image = center_image[origin_index]
        center_image = np.expand_dims(center_image, axis=0)
        center_image = np.expand_dims(center_image, axis=-1)

        segmentation_image = segmentation_image[origin_index]
        segmentation_image = np.expand_dims(segmentation_image, axis=0)
        segmentation_image = np.expand_dims(segmentation_image, axis=-1)

        one_center = one_center[origin_index]
        one_center = np.squeeze(one_center)
        one_segmentation = one_segmentation[origin_index]
        one_segmentation = np.squeeze(one_segmentation)

        color_label = color_label[origin_index]

        result_image_list = []
        total_image = None

        iou_list_of_centers = []
        dice_list_of_ceenters = []

        # NOTICE : for centers.
        for i in range(len(one_center)):
            one_center_image = one_center[i]
            one_center_image = np.expand_dims(one_center_image, axis=0)
            one_center_image = np.expand_dims(one_center_image, axis=-1)
            one_segmentation_image = one_segmentation[i]
            one_segmentation_image = np.expand_dims(one_segmentation_image, axis=0)
            one_segmentation_image = np.expand_dims(one_segmentation_image, axis=-1)

            a = sess.run(
                out_match,
                feed_dict={
                    input: origin_image,
                    label_center: center_image,
                    label_segmentation: segmentation_image,
                    individual_points: one_center_image,
                    label_match: one_segmentation_image,
                    training: False})

            a = np.reshape(a, [512, 512])

            # NOTICE: threshold : 0.5
            a[a <= 0.5] = 0
            a[a > 0.5] = 1

            # find noise region: ############################################
            # 같은 네트워크에서 나온 노이즈는 같다,
            c_1 = one_center[0]
            c_1 = np.expand_dims(c_1, axis=0)
            c_1 = np.expand_dims(c_1, axis=-1)
            s_1 = one_segmentation[0]
            s_1 = np.expand_dims(s_1, axis=0)
            s_1 = np.expand_dims(s_1, axis=-1)

            a_1 = sess.run(
                out_match,
                feed_dict={
                    input: origin_image,
                    label_center: center_image,
                    label_segmentation: segmentation_image,
                    individual_points: c_1,
                    label_match: s_1,
                    training: False})

            a_1 = np.reshape(a_1, [512, 512])

            c_2 = one_center[1]
            c_2 = np.expand_dims(c_2, axis=0)
            c_2 = np.expand_dims(c_2, axis=-1)
            s_2 = one_segmentation[1]
            s_2 = np.expand_dims(s_2, axis=0)
            s_2 = np.expand_dims(s_2, axis=-1)

            a_2 = sess.run(
                out_match,
                feed_dict={
                    input: origin_image,
                    label_center: center_image,
                    label_segmentation: segmentation_image,
                    individual_points: c_2,
                    label_match: s_2,
                    training: False})

            a_2 = np.reshape(a_2, [512, 512])

            # find noise region: #############################################3
            noise_region_index = np.where((a_1 == a_2) & (a_1 == 1))
            # remove noise
            a[noise_region_index] = 0

            if len(result_image_list) > 0:
                a_prev = result_image_list[-1]

                # NOTICE: delete new overlap region.
                total_image[np.where(total_image == a)] = 0
                # gray coloring.
                total_image[np.where(a == 1)] = grayscale_list[i]
            else:
                total_image = a

            result_image_list.append(a)

            plt.imshow(a)
            plt.show()

            b = np.squeeze(one_segmentation_image)
            b[b > 0.5] = 1
            plt.imshow(b)
            plt.show()

            only_a = a.copy()
            only_a[np.where(a == b)] = 0
            only_a[np.where(b == 1)] = 0
            together = a.copy()
            together[np.where(b == 0)] = 0
            only_target = b.copy()
            only_target[np.where(a == b)] = 0

            iou = np.count_nonzero(together) / (np.count_nonzero(only_a) + np.count_nonzero(only_target) + np.count_nonzero(together))
            dice = (2 * np.count_nonzero(together)) / (np.count_nonzero(only_a) + np.count_nonzero(only_target) + (2 * np.count_nonzero(together)))

            iou_list_of_centers.append(iou)
            dice_list_of_ceenters.append(dice)

            print("one instance iou ; {}".format(iou))
            print("one instance dice(%) ; {}%".format(dice))

        total_iou = np.mean(iou_list_of_centers)
        print("total_iou : {}".format(total_iou))
        IOU_LIST.append(total_iou)
        plt.imshow(total_image, cmap="gray")
        plt.show()

        # save result.
        imsave("/data1/LJH/Dot_To_Mask_instance_segmentation/saved_models/test_01_result/plant_{}_predict.png".format(origin_index), total_image)
        imsave("/data1/LJH/Dot_To_Mask_instance_segmentation/saved_models/test_01_result/plant_{}_target.png".format(origin_index), color_label)

        print(len(result_image_list))

    model_iou = np.mean(IOU_LIST)
    print("###### total model IOU : {} #######".format(model_iou))
