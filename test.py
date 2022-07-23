
import config


file_name = 'test12345'
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
data_dir = '../yolov3_0/data_animals' # TODO move data in separate folder

from yolo_utils import (
    loadDictoniaries, 
    loadImagesAndTargets, 
    best_anchors,
    convertToTargetFormat,
    convertToTargetFormata,
    reshapeImages
    )

images_dict, labels_dict = loadDictoniaries(data_dir)

x, y = loadImagesAndTargets(images_dict, labels_dict)

# Prepare Images
# Reshape
x_reshaped = reshapeImages(x,img_size)          
#%%
# Convert y in pre-target shape. Pre-target shape is needed for split
# Format: targets[image][grid_scale][grid,grid,(c,x,y,w,h,(classes))*anchors]
y_targets = convertToTargetFormat(y, anchors, grid_list, num_classes)

# Split into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_reshaped, 
                                                    y_targets, 
                                                    random_state=0)

# # TODO testing output
# x_train, y_train = x_train[:10], y_train[:10]
# x_test, y_test = x_test[:3], y_test[:3]

# Convert into final target shape and format(numpy array)
# Format: targets[grid_scale][image][grid,grid,(c,x,y,w,h,(classes))*anchors]
y1,y2,y3 = list(map(list, zip(*y_train))) # TODO change in loading data

import numpy as np
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y_train = [y1,y2,y3]

# Test set
y1,y2,y3 = list(map(list, zip(*y_test))) # TODO change in loading data

y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y_test = [y1,y2,y3]

# scale anchors
scaled_anchors = []
for i,_ in enumerate(grid_list):
    scaled_anchors.append(np.array(anchors[i])*grid_list[i])
scaled_anchors = np.array(scaled_anchors)

# #%%
# from draw_functions import draw_check

# draw_check(x_test,y_test, 20)
#%% Model

from model import (
build_model, yoloLoss, yoloMetrics, testLoss, l_obj, l_noobj, l_box, l_class,
CustomCallback
    )
import tensorflow as tf

import time
# start_time = time.time()
#
yolo = build_model(num_classes)
g_ids = list(range(len(grid_list)))
# loss_function = [yoloLoss(anchors, weights, g_id, num_classes) for g_id in g_ids]
loss_function = [yoloLoss(scaled_anchors, weights, 0, num_classes),
                 yoloLoss(scaled_anchors, weights, 1, num_classes),
                 yoloLoss(scaled_anchors, weights, 2, num_classes)]
# metrics = yoloMetrics(anchors, weights, grid_list, num_classes)
metrics = l_obj(anchors, weights, 0, num_classes)

yolo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
             loss=loss_function,
             # metrics=metrics
             )


history = yolo.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                   validation_data=(x_test, y_test),
                    callbacks=[
                        CustomCallback(
                            x_test[0], y_test[0],
                            scaled_anchors,
                            yolo, 
                            "callback"
                            )
                        ]
                   ) 