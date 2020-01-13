from tools.Handler import Handler
import matplotlib.pyplot as plt
import numpy as np
import pickle
import skimage.draw as draw
from PIL import Image, ImageDraw, ImageFont
import math


class DataHandler(Handler):

    def __init__(self, oval_ratio):

        self._i = 0

        # read data.
        super().__init__("title")
        self.all_classes = ["raccoon"]

        self.color_map = ["#12b33d"]

        # load data.
        with open('/home/user01/data_ssd/LeeJongHyeok/Dot_To_Mask_instance_segmentation/racoon/dat.pickle', 'rb') as f:
            data = pickle.load(f)
        self.names = data["names"]
        self.sizes = data["sizes"]
        self.box_list = data["box_list"]
        self.box_class = data["box_class"]
        self.images = []
        self.images_label_point_center_1_by_1 = []
        self.images_label_point_center_2_by_2 = []
        self.images_label_point_center_4_by_4 = []
        self.images_label_point_center_8_by_8 = []

        # load images
        for name in self.names:
            image = Image.open(name[0])
            self.images.append(np.array(image) / 255)

        # create center points images.
        for index, box_data in enumerate(self.box_list):
            img_height = int(self.sizes[index][0])
            img_width = int(self.sizes[index][1])

            coordinates = box_data[4]  # (w, h)

            center_points_image = np.zeros(shape=[img_height, img_width])

            x = int(coordinates[0])  # w
            y = int(coordinates[1])  # h

            center_points_image[y, x] = 1

            self.images_label_point_center_1_by_1.append(center_points_image)

            # for /2
            center_points_image = np.zeros(shape=[math.ceil(img_height / 2), math.ceil(img_width / 2)])

            x = (x - 1) // 2
            y = (y - 1) // 2
            center_points_image[y, x] = 1

            self.images_label_point_center_2_by_2.append(center_points_image)

            # for /4
            center_points_image = np.zeros(shape=[math.ceil(img_height / 4), math.ceil(img_width / 4)])

            x = (x - 1) // 4
            y = (y - 1) // 4
            center_points_image[y, x] = 1

            self.images_label_point_center_4_by_4.append(center_points_image)

            # for /8
            center_points_image = np.zeros(shape=[math.ceil(img_height / 8), math.ceil(img_width / 8)])

            x = (x - 1) // 8
            y = (y - 1) // 8
            center_points_image[y, x] = 1

            self.images_label_point_center_8_by_8.append(center_points_image)

            """           
            original_size: { h, w}
            box list: {start_h, start_w, height, width},
            box class: {class_name}
            _Center : {w, h}
            """

        self.data_size = len(self.names)
        print("Total data length : {}".format(self.data_size))  # 185 train 155, test:30

        self.train_images = self.images[:154]
        self.train_images_size = self.sizes[:154]
        self.train_images_box_list = self.box_list[:154]
        self.train_images_box_class = self.box_class[:154]
        self.train_images_label_point_center_1_by_1 = self.images_label_point_center_1_by_1[:154]
        self.train_images_label_point_center_2_by_2 = self.images_label_point_center_2_by_2[:154]
        self.train_images_label_point_center_4_by_4 = self.images_label_point_center_4_by_4[:154]
        self.train_images_label_point_center_8_by_8 = self.images_label_point_center_8_by_8[:154]

        self.test_images = self.images[155:]
        self.test_images_size = self.sizes[155:]
        self.test_images_box_list = self.box_list[155:]
        self.test_images_box_class = self.box_class[155:]
        self.test_images_label_point_center_1_by_1 = self.images_label_point_center_1_by_1[155:]
        self.test_images_label_point_center_2_by_2 = self.images_label_point_center_2_by_2[155:]
        self.test_images_label_point_center_4_by_4 = self.images_label_point_center_4_by_4[155:]
        self.test_images_label_point_center_8_by_8 = self.images_label_point_center_8_by_8[155:]

    def run(self):
        pass

    # print next batch.
    def nextBatch(self, batch_size):
        train_images = self.train_images[self._i: self._i + batch_size]
        train_images1by1 = self.train_images_label_point_center_1_by_1[self._i: self._i + batch_size]
        train_images2by2 = self.train_images_label_point_center_2_by_2[self._i: self._i + batch_size]
        train_images4by4 = self.train_images_label_point_center_4_by_4[self._i: self._i + batch_size]
        train_images8by8 = self.train_images_label_point_center_8_by_8[self._i: self._i + batch_size]

        self._i = (self._i + batch_size) % len(self.train_images)

        return train_images, train_images1by1, train_images2by2, train_images4by4, train_images8by8

    def plot_rect_on_image(self, image, rects, line_width=3):
        """
        :param image: 1 image.
        :param rects: rect list (start_w, start_h, width, height, center)
        :return:
        """
        plotting = image.copy()
        color_r = int(self.color_map[0][1:3], 16)
        color_g = int(self.color_map[0][3:5], 16)
        color_b = int(self.color_map[0][5:7], 16)

        # plot rect.
        rect = rects
        for offset in range(line_width):
            # rr, cc = line(start_h, start_w, end_h, end_w)
            start_h = int(rect[0])
            start_w = int(rect[1])
            end_h = int(rect[0]) + int(rect[2])
            end_w = int(rect[1]) + int(rect[3])
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
            font = ImageFont.truetype(
                '/home/user01/data_ssd/LeeJongHyeok/Dot_To_Mask_instance_segmentation/asset/NANUMSQUAREB.TTF', size=20)
            ImageDraw.Draw(plotting).rectangle((int(rect[1]), int(rect[0]), int(rect[1]) + 100, int(rect[0]) + 30),
                                               fill=self.color_map[0])

            print("class : {}".format(self.all_classes[0]))

            ImageDraw.Draw(plotting).text((int(rect[1]) + line_width + 3, int(rect[0]) + line_width + 3),
                                          self.all_classes[0],
                                          fill="#ffffff", font=font)
            plotting = np.array(plotting)
        plt.imshow(plotting)
        plt.show()

# # test darw regions.
# handler = DataHandler(3)
# for i in range(100):
#     handler.plot_rect_on_image(handler.images[i], handler.box_list[i], line_width=3)
