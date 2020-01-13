import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

input_test = tf.constant([4., 2., 2., 3., 3., 5., 7., 8., 9., 4., 2., 3., 2., 1.])
label_test = tf.constant([1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.])


# def get_focal_loss(input, label):
#     label_count = tf.constant(0)
#     loss_sum = tf.constant(0.)
#
#     def cond(i, loss_sum):
#         print(tf.size(input))
#         return tf.less(i, tf.size(input))
#
#     def body(i, loss_sum):
#         print("body_start")
#
#         tensor_1 = tf.squeeze(tf.gather(input, i))
#
#         elements = input
#
#         def get_loss_each_element(element):
#             loss = tf.cond(tf.equal(tensor_1, tf.constant(1.)),
#                            lambda: tf.square(tf.constant(1.) - element) * tf.log(element),
#                            lambda: tf.pow(tf.constant(1.) - tensor_1, tf.constant(4.)) * tf.square(
#                                tf.log(element)) * tf.log(tf.constant(1.) - element))
#
#             return loss
#
#         loss_list = tf.map_fn(get_loss_each_element, elements)
#
#         loss_sum = tf.add(loss_sum, tf.reduce_sum(loss_list))
#
#         return tf.add(i, 1), loss_sum
#
#     result = tf.while_loop(cond, body, [label_count, loss_sum])
#
#     nums_1 = tf.cast(tf.count_nonzero(label), tf.float32)
#     loss_sum = (-1. * tf.reciprocal(nums_1)) * result[1]  # result[1] :  loss_sum
#
#     return result[0], loss_sum
def get_focal_loss_v2(input, label):
    loss_list = tf.where(condition=tf.cast(label, tf.bool),
                         x=tf.square(tf.constant(1.) - input) * tf.log(input),
                         y=tf.square(input) * tf.log(tf.constant(1.) - input))

    result = tf.reduce_sum(loss_list)

    nums_1 = tf.cast(tf.count_nonzero(label), tf.float32)
    loss_sum = (-1. * tf.reciprocal(nums_1)) * result

    return loss_sum


with tf.Session() as sess:

    out_test = tf.nn.softmax(input_test)

    print(sess.run(get_focal_loss_v2(out_test, label_test)))

    loss_list = tf.where(condition=tf.cast(label_test, tf.bool),
                         x=tf.square(tf.constant(1.) - out_test) * tf.log(out_test),
                         y=tf.square(tf.log(out_test)) * tf.log(tf.constant(1.) - out_test))

    loss_list = sess.run(loss_list)
