import numpy as np
from scipy.misc import imread
from os import listdir
from PIL import Image, ImageDraw


class DataManager:

    def __init__(self):
        self.batch_flag = 0

        self.data_x = []
        self.data_center = []
        self.data_segmentation = []

        # notice : load data.
        path = "/data1/LJH/Dot_To_Mask_instance_segmentation/A1/"

        dir_list = listdir(path)
        dir_list = sorted(dir_list)

        for file in dir_list:

            file_path = path + file

            if "_centers" in file:
                print(file)
                img_centers = imread(file_path, mode="L")
                coordinates = np.where(img_centers)

                pil_image_centers = Image.fromarray(np.zeros(np.shape(img_centers)))
                draw_circle = ImageDraw.Draw(pil_image_centers)
                for i in range(len(coordinates[0])):
                    x = coordinates[0][i]
                    y = coordinates[1][i]

                    # add circle.
                    draw_circle.ellipse((y - 5, x - 5, y + 5, x + 5), fill='white')
                self.data_center.append(np.array(pil_image_centers))

            if "_fg" in file:
                self.data_segmentation.append(imread(file_path, mode='L'))

            if "_rgb" in file:
                print(file)
                self.data_x.append(imread(file_path, mode='RGB'))

    def next_batch(self, total_images, total_labels1, total_labels2, batch_size):

        sub_batch_x, sub_batch_y1, sub_batch_y2 = total_images[
                                                  self.batch_flag: self.batch_flag + batch_size], total_labels1[
                                                                                                  self.batch_flag:self.batch_flag + batch_size], total_labels2[
                                                                                                                                                 self.batch_flag:self.batch_flag + batch_size]
        self.batch_flag = (self.batch_flag + batch_size) % len(total_images)
        return sub_batch_x, sub_batch_y1, sub_batch_y2
