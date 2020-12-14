import numpy as np
from scipy.misc import imread, imsave
from os import listdir
from PIL import Image, ImageDraw
from tools.CenterTools import get_centers, get_areas_from_center


class DataManager:

    def __init__(self, ratio=0.9):
        self.batch_flag = 0

        self.data_x = []
        self.data_center = []
        self.data_one_center_img = []
        self.data_one_center_list = []
        self.data_one_center_label = []
        self.data_segmentation = []
        self.data_color_label = []

        # notice : load data.
        path = "/data1/LJH/Dot_To_Mask_instance_segmentation/A1/"

        dir_list = listdir(path)
        dir_list = sorted(dir_list)

        for file in dir_list:
            file_path = path + file

            if "_centers" in file:
                origin_path = file_path
                print(file)
                img_centers = imread(file_path, mode="L")
                coordinates = np.where(img_centers)

                pil_image_centers = Image.fromarray(np.zeros(np.shape(img_centers)))
                draw_circle = ImageDraw.Draw(pil_image_centers)
                for i in range(len(coordinates[0])):
                    x = coordinates[0][i]
                    y = coordinates[1][i]

                    # add circle.
                    draw_circle.ellipse((y - 6, x - 6, y + 6, x + 6), fill='white')
                self.data_center.append(np.array(pil_image_centers) / 255)
                # imsave(file_path.replace("_centers.png", "") + "_hd.png", np.array(pil_image_centers))
                imgs, indices = get_centers(file_path.replace("_centers.png", "") + "_hd.png")

                self.data_one_center_img.append(imgs)
                self.data_one_center_list.append(indices)

                # save label images.
                each_images = get_areas_from_center(origin_path.replace("_centers.png", "") + "_label.png", indices)

                self.data_one_center_label.append(each_images)

            if "_fg" in file:
                self.data_segmentation.append(imread(file_path, mode='L') / 255)

            if "_rgb" in file:
                print(file)
                self.data_x.append(imread(file_path, mode='RGB'))

            if file.endswith("_label.png"):
                self.data_color_label.append(imread(file_path, mode='RGB'))

        # divide data set.

        train_dataset_size = int(ratio * len(self.data_x))
        test_dataset_size = len(self.data_x) - train_dataset_size

        print("train_dataset_size :{} , test_dataset_size :{}".format(train_dataset_size, test_dataset_size))

        self.train_data_x = self.data_x[:train_dataset_size]
        self.train_data_center = self.data_center[:train_dataset_size]
        self.train_data_one_center_img = self.data_one_center_img[:train_dataset_size]
        self.train_data_one_center_list = self.data_one_center_list[:train_dataset_size]
        self.train_data_one_center_label = self.data_one_center_label[:train_dataset_size]
        self.train_data_segmentation = self.data_segmentation[:train_dataset_size]

        self.test_data_x = self.data_x[train_dataset_size:]
        self.test_data_center = self.data_center[train_dataset_size:]
        self.test_data_one_center_img = self.data_one_center_img[train_dataset_size:]
        self.test_data_one_center_list = self.data_one_center_list[train_dataset_size:]
        self.test_data_one_center_label = self.data_one_center_label[train_dataset_size:]
        self.test_data_segmentation = self.data_segmentation[train_dataset_size:]

        self.test_data_color_label = self.data_color_label[train_dataset_size:]

    def next_batch(self, total_images, total_labels1, total_labels2, batch_size):

        sub_batch_x, sub_batch_y1, sub_batch_y2 = total_images[self.batch_flag: self.batch_flag + batch_size], \
                                                  total_labels1[self.batch_flag:self.batch_flag + batch_size], \
                                                  total_labels2[self.batch_flag:self.batch_flag + batch_size]
        self.batch_flag = (self.batch_flag + batch_size) % len(total_images)
        return sub_batch_x, sub_batch_y1, sub_batch_y2

    def next_batch_with_each_center(self, batch_size):

        origin_image, one_center, segmentation_image, one_segmentation, center_image = self.train_data_x[
                                                                                       self.batch_flag: self.batch_flag + batch_size], \
                                                                                       self.train_data_one_center_img[
                                                                                       self.batch_flag:self.batch_flag + batch_size], \
                                                                                       self.train_data_segmentation[
                                                                                       self.batch_flag:self.batch_flag + batch_size], \
                                                                                       self.train_data_one_center_label[
                                                                                       self.batch_flag:self.batch_flag + batch_size], \
                                                                                       self.train_data_center[
                                                                                       self.batch_flag:self.batch_flag + batch_size]

        self.batch_flag = (self.batch_flag + batch_size) % len(self.data_x)
        return origin_image, center_image, segmentation_image, one_center, one_segmentation

    def get_test_data(self):
        return self.test_data_x, self.test_data_center, self.test_data_segmentation, self.test_data_one_center_img, self.test_data_one_center_label, self.test_data_color_label

# db = DataManager()
# print("")
