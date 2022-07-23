# =============================================================================
# In the conversion of specific target shape the coordinates of the right
# cell location is grid[y][x]. Like in this repo:
# https://github.com/ethanyanjiali/deep-vision/blob/527bb3da655ac6245568942e252e27c61d0b4ca2/YOLO/tensorflow/preprocess.py
# This might be important in loading weights from the paper, because a cell
# location with grid[x][y] might not work when using transfer learning
# output shape of model: [BS, Cell_y, Cell_x, Anchor, (x,y,w,h,conf,...classes)]
# =============================================================================

import config

file_name = 'weight'
img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.TEST_EPOCHS
batch_size=config.TEST_BATCH_SIZE
learning_rate = config.LEARNING_RATE # TODO

# =============================================================================
# load data
# =============================================================================
# data_dir = '../yolov3_0/data_animals'
data_dir = '../yolov3_0/data_example2'


from yolo_utils import (
    loadDictoniaries, 
    loadImagesAndTargets, 
    convertToTargetFormat,
    reshapeImages
    )

images_dict, labels_dict = loadDictoniaries(data_dir)

x, y = loadImagesAndTargets(images_dict, labels_dict)

# Prepare Images
# Reshape
x_reshaped = reshapeImages(x,img_size)          

# Convert y in pre-target shape. Pre-target shape is needed for split
# Format: targets[image][grid_scale][grid,grid,(c,x,y,w,h,(classes))*anchors]
y_targets = convertToTargetFormat(y, anchors, grid_list, num_classes)
#%%

rhino_x = y_targets[4][0][:,:,1,1]
rhino_y = y_targets[4][0][:,:,1,2]

from yolo_utils import _meshgrid
import numpy as np
rhino_x1 = rhino_x * 13 - np.array(_meshgrid(13,13)[0])
rhino_y1 = rhino_y * 13 - np.array(_meshgrid(13,13)[1])

rhino = np.stack((rhino_x1, rhino_y1), -1)

# print('My _meshgrid')
print('My _meshgrid')
print(rhino[4,9])
print(rhino[4,9]-[0.38622599999999974, 0.9736519999999995] == [0.,0.])
# prijnt('Other _meshgrid')
print('Other _meshgrid')
print(rhino[9,4])
print(rhino[9,4]-[0.38622599999999974, 0.9736519999999995] == [0.,0.])

