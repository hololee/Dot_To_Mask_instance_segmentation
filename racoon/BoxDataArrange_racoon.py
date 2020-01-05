import os
import pickle
import numpy as np
import xml.etree.ElementTree as elemTree

all_classes = ["racoon"]
offset = 1  # start of x, y coordinate = (1, 1)

"""
- data structure
{
file_name: {n, file name}
original_size: {n , h, w}
box list  : {n, start_h ,start_w , height, width}, 
box class : {n, class_name} 
}

"""

file_names = []
original_sizes = []
box_lists = []
box_classes = []

project_dir = "/home/user01/data_ssd/LeeJongHyeok/Dot_To_Mask_instance_segmentation/racoon/"
dir = project_dir

# get file list.
file_dir_list_all = [dir + x for x in os.listdir(dir)]
file_dir_list = []
for data in file_dir_list_all:
    if data.endswith("xml"):
        file_dir_list.append(data)
file_dir_list.sort()

# read and add data.
for file_dir in file_dir_list:
    # read xml.
    tree = elemTree.parse(file_dir)
    _file_name = tree.find("./filename").text
    _size = tree.find("./size")

    _width = _size.find("./width").text
    _height = int(_size.find("./height").text)
    _depth = int(_size.find("./depth").text)

    _object = tree.find("./object")
    _bndbox = _object.find("./bndbox")

    file_names.append(project_dir + _file_name)
    original_sizes.append([_height, _width])
    _Center = ((int(_bndbox.find("./xmin").text) + int(_bndbox.find("./xmax").text)) / 2,
               (int(_bndbox.find("./ymin").text) + int(_bndbox.find("./ymax").text)) / 2)
    box_lists.append([int(_bndbox.find("./ymin").text), int(_bndbox.find("./xmin").text),
                      int(_bndbox.find("./ymax").text) - int(_bndbox.find("./ymin").text),
                      int(_bndbox.find("./xmax").text) - int(_bndbox.find("./xmin").text), _Center])
    box_classes.append("raccoon")

file_names = np.array(file_names)
file_names = np.reshape(file_names, [-1, 1])
original_sizes = np.array(original_sizes)
box_lists = np.array(box_lists)
box_classes = np.array(box_classes)
box_classes = np.reshape(box_classes, [-1, 1])

data = {"names": file_names, "sizes": original_sizes, "box_list": box_lists, "box_class": box_classes}
with open(project_dir + 'dat.pickle', 'wb') as f:
    pickle.dump(data, f)

print("finish exit.")
