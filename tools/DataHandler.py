from tools.Handler import Handler
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.misc
import skimage.draw as draw
from PIL import Image, ImageDraw, ImageFont


class DataHandler(Handler):

    def __init__(self, title):

        self._i = 0

        # read data.
        super().__init__(title)
        self.all_classes = ["bicycle",
                            "bus",
                            "car",
                            "cat",
                            "cow",
                            "dog",
                            "horse",
                            "motorbike",
                            "person",
                            "sheep"]

        self.color_map = ["#12b33d",
                          "#b32a12",
                          "#b37512",
                          "#b0b312",
                          "#12b380",
                          "#12a8b3",
                          "#126bb3",
                          "#1235b3",
                          "#6212b3",
                          "#a312b3"]

        # load data.
        with open('/data1/LJH/faster_rcnn_implement/VOC2006/dat.pickle', 'rb') as f:
            data = pickle.load(f)
        self.names = data["names"]
        self.sizes = data["sizes"]
        self.box_list = data["box_list"]
        self.box_class = data["box_class"]
        self.images = []

        for name in self.names:
            image = Image.open(name)
            self.images.append(np.array(image))

        self.data_size = len(self.names)
        print("Total data length : {}".format(self.data_size))  # 2618 train 2118, test:500

        self.train_images = self.images[:2117]
        self.train_images_size = self.sizes[:2117]
        self.train_images_box_list = self.box_list[:2117]
        self.train_images_box_class = self.box_class[:2117]

        self.test_images = self.images[2118:]
        self.test_images_size = self.images[2118:]
        self.test_images_box_list = self.images[2118:]
        self.test_images_box_class = self.images[2118:]

    def run(self):
        pass

    # print next batch.
    def nextBatch(self, batch_size):

        train_images = self.train_images[self._i:self._i + batch_size]
        train_labels = np.zeros(shape=[np.shape(train_images)[0], np.shape(train_images)[1], 2])
        # channel 0 :  width
        train_labels[self.train_images_box_list[self._i: self._i + batch_size][4][0],
                     self.train_images_box_list[self._i: self._i + batch_size][4][1], 0] = \
        self.train_images_box_list[self._i: self._i + batch_size][2]
        # channel 1 :  height
        train_labels[self.train_images_box_list[self._i: self._i + batch_size][4][1],
                     self.train_images_box_list[self._i: self._i + batch_size][4][1], 1] = \
        self.train_images_box_list[self._i: self._i + batch_size][3]

        self._i = (self._i + batch_size) % len(self.train_images)

        return train_images, train_labels

    def plot_rect_on_image(self, image, rects, classes, line_width=3):
        """
        :param image: 1 image.
        :param rects: rect list (start_w, start_h, width, height, center)
        :return:
        """
        plotting = image.copy()

        for _index, _class in enumerate(classes):

            color_r = int(self.color_map[int(np.argmax(_class))][1:3], 16)
            color_g = int(self.color_map[int(np.argmax(_class))][3:5], 16)
            color_b = int(self.color_map[int(np.argmax(_class))][5:7], 16)

            # plot rect.
            rect = rects[_index]
            for offset in range(line_width):
                # rr, cc = line(start_h, start_w, end_h, end_w)
                start_h = rect[0]
                start_w = rect[1]
                end_h = rect[0] + rect[2]
                end_w = rect[1] + rect[3]
                offset = -offset
                rr_top, cc_top = draw.line(start_h - offset, start_w - offset, start_h - offset, end_w + offset)
                rr_bot, cc_bot = draw.line(end_h + offset, start_w - offset, end_h + offset, end_w + offset)
                rr_left, cc_left = draw.line(start_h - offset, start_w - offset, end_h + offset, start_w - offset)
                rr_right, cc_right = draw.line(start_h - offset, end_w + offset, end_h + offset, end_w + offset)

                try:
                    # draw top.
                    plotting[rr_top, cc_top, 0] = color_r
                    plotting[rr_top, cc_top, 1] = color_g
                    plotting[rr_top, cc_top, 2] = color_b
                    # draw bot
                    plotting[rr_bot, cc_bot, 0] = color_r
                    plotting[rr_bot, cc_bot, 1] = color_g
                    plotting[rr_bot, cc_bot, 2] = color_b
                    # draw left
                    plotting[rr_left, cc_left, 0] = color_r
                    plotting[rr_left, cc_left, 1] = color_g
                    plotting[rr_left, cc_left, 2] = color_b
                    # draw right
                    plotting[rr_right, cc_right, 0] = color_r
                    plotting[rr_right, cc_right, 1] = color_g
                    plotting[rr_right, cc_right, 2] = color_b
                except:
                    print("out of bounds.")

            plotting = Image.fromarray(plotting, 'RGB')
            font = ImageFont.truetype('/data1/LJH/faster_rcnn_implement/asset/NANUMSQUAREB.TTF', size=20)
            ImageDraw.Draw(plotting).rectangle((rect[1], rect[0], rect[1] + 70, rect[0] + 30),
                                               fill=self.color_map[int(np.argmax(_class))])

            print("class : {}".format(self.all_classes[int(np.argmax(_class))]))

            ImageDraw.Draw(plotting).text((rect[1] + line_width + 3, rect[0] + line_width + 3),
                                          self.all_classes[int(np.argmax(_class))],
                                          fill="#ffffff", font=font)
            plotting = np.array(plotting)
        plt.imshow(plotting)
        plt.show()

# # test darw regions.
# handler = DataHandler("data")
# for i in range(100):
#     handler.plot_rect_on_image(handler.images[i], handler.box_list[i], handler.box_class[i], line_width=3)
