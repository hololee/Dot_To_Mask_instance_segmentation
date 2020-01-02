"""This script extract the Box data from VOC2006/Annotations."""
import os
import pickle
import numpy as np

all_classes = ["bicycle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "person", "sheep"]
offset = 1  # start of x, y coordinate = (1, 1)

"""
- data structure
{
file_name: {n, file name}
original_size: {n , h, w}
box list  : {n, m, start_h ,start_w , height, width}, 
box class : {n, m, class_name} 
}

"""

file_names = []
original_sizes = []
box_lists = []
box_classes = []

project_dir = "/data1/LJH/faster_rcnn_implement/"
dir = project_dir + "VOC2006/Annotations/"

# get file list.
file_dir_list = [dir + x for x in os.listdir(dir)]
file_dir_list.sort()

# read and add data.
for file_dir in file_dir_list:
    file = open(file_dir, 'r')
    lines = file.readlines()

    box_list = []
    box_class = []

    for line in lines:
        if line.startswith("Image filename"):
            title = line.split("\"")[1]
            file_names.append(project_dir + title)
            print(title)
        elif line.startswith("Image size"):
            w = int(line.split(":")[1].split("x")[0])
            h = int(line.split(":")[1].split("x")[1])
            original_sizes.append([h, w])
        elif line.startswith("Original label"):
            for i in all_classes:
                class_raw_name = line.split(":")[1].replace("\"", "").replace("\n", "").strip()
                if i in class_raw_name:
                    onehot = np.zeros(10)
                    onehot[all_classes.index(i)] = 1
                    box_class.append(onehot)
        elif line.startswith("Bounding box for object"):
            Xmin = int(line.split(":")[1].split("-")[0].replace("(", "").replace(")", "").split(",")[0])
            Ymin = int(line.split(":")[1].split("-")[0].replace("(", "").replace(")", "").split(",")[1])
            Xmax = int(line.split(":")[1].split("-")[1].replace("(", "").replace(")", "").split(",")[0])
            Ymax = int(line.split(":")[1].split("-")[1].replace("(", "").replace(")", "").split(",")[1])
            Center = (Xmin + Xmax, Ymin + Ymax)
            box_list.append([Ymin, Xmin, Ymax - Ymin, Xmax - Xmin, Center])

    box_lists.append(box_list)
    box_classes.append(box_class)

file_names = np.array(file_names)
original_sizes = np.array(original_sizes)
box_lists = np.array(box_lists)
box_classes = np.array(box_classes)

data = {"names": file_names, "sizes": original_sizes, "box_list": box_lists, "box_class": box_classes}
with open('/data1/LJH/faster_rcnn_implement/VOC2006/dat.pickle', 'wb') as f:
    pickle.dump(data, f)

print("finish exit.")
