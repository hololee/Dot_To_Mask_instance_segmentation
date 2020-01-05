from tools.Handler import Handler
import matplotlib.pyplot as plt
import numpy as np
import pickle
import skimage.draw as draw
from PIL import Image, ImageDraw, ImageFont


class DataHandler(Handler):

    def __init__(self, round_size):

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
        self.images_label_point_center = []
        self.images_label_object_detection = []

        # load images
        for name in self.names:
            image = Image.open(name[0])
            self.images.append(np.array(image))

        # create center points images.
        for index, box_data in enumerate(self.box_list):
            coordinates = box_data[4]  # (w, h)

            pil_image_centers = Image.fromarray(np.zeros(shape=[int(self.sizes[index][0]), int(self.sizes[index][1])]))
            draw_circle = ImageDraw.Draw(pil_image_centers)

            x = int(coordinates[0]) # w
            y = int(coordinates[1]) # h

            # add circle.
            draw_circle.ellipse((x - round_size, y - round_size, x + round_size, y + round_size), fill='white')
            self.images_label_point_center.append(np.array(pil_image_centers))

            """           
            original_size: { h, w}
            box list: {start_h, start_w, height, width},
            box class: {class_name}
            _Center : {w, h}
            """

            #  create objective output.
            objective_image = np.zeros(shape=[int(self.sizes[index][0]), int(self.sizes[index][1]), 2])
            objective_image[y, x, 0] = self.box_list[index][2]  # set height
            objective_image[y, x, 1] = self.box_list[index][3]  # set width
            self.images_label_object_detection.append(objective_image)

        self.data_size = len(self.names)
        print("Total data length : {}".format(self.data_size))  # 200 train 160, test:40

        self.train_images = self.images[:159]
        self.train_images_size = self.sizes[:159]
        self.train_images_box_list = self.box_list[:159]
        self.train_images_box_class = self.box_class[:159]
        self.train_images_label_point_center = self.images_label_point_center[:159]
        self.train_images_label_object_detection = self.images_label_object_detection[:159]

        self.test_images = self.images[160:]
        self.test_images_size = self.sizes[160:]
        self.test_images_box_list = self.box_list[160:]
        self.test_images_box_class = self.box_class[160:]
        self.test_images_label_point_center = self.images_label_point_center[160:]
        self.test_images_label_object_detection = self.images_label_object_detection[160:]

    def run(self):
        pass

    # print next batch.
    def nextBatch(self, batch_size):
        train_images = self.train_images[self._i: self._i + batch_size]
        train_labels_point_seg = self.train_images_label_point_center[self._i: self._i + batch_size]
        train_labels_object_Det = self.train_images_label_object_detection[self._i: self._i + batch_size]
        self._i = (self._i + batch_size) % len(self.train_images)

        return train_images, train_labels_point_seg, train_labels_object_Det

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

#
# # test darw regions.
# handler = DataHandler("data")
# for i in range(100):
#     handler.plot_rect_on_image(handler.images[i], handler.box_list[i], line_width=3)
