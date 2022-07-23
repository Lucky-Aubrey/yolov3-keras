
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

# Split into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_reshaped, 
                                                    y_targets, 
                                                    random_state=0)

# TODO testing output
x_train, y_train = x_train[:10], y_train[:10]
x_test, y_test = x_test[:3], y_test[:3]

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
#%% Test github_load_dataset voc2012
from github_load_datasets import get_voc2012_datasets

train_dataset, val_dataset = get_voc2012_datasets()
num_classes = 20
#%% Model

from model import (
# build_model,
yoloLoss, CustomCallback, ModelOutput
    )
# Test named model for weight converter
from weight_converter_model import build_model
import tensorflow as tf


yolo = build_model(num_classes)
coco_yolo = build_model(80)

coco_yolo.load_weights('weights/yolov3_coco.tf')

yolo.get_layer('yolo_darknet').set_weights(
    coco_yolo.get_layer('yolo_darknet').get_weights())

loss_function = [yoloLoss(anchors[0], weights, num_classes),
                 yoloLoss(anchors[1], weights, num_classes),
                 yoloLoss(anchors[2], weights, num_classes)]

yolo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
             loss=loss_function)

# history = yolo.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
#                     # validation_data = [x_train, y_train]
#                     # validation_data = [x_val, y_val]
#                     validation_data=(x_test, y_test),
#                     # callbacks = [callback]
#                     ) 
# Use voc2012 dataset
history = yolo.fit(train_dataset, batch_size=batch_size, epochs=epochs,
                    # validation_data = [x_train, y_train]
                    # validation_data = [x_val, y_val]
                    validation_data=val_dataset,
                    # callbacks = [callback]
                    ) 

# yolo.save_weights('weights/'+file_name+'.h5')
# np.save('history/'+file_name+'.npy', history.history)

#%% Detect
import matplotlib.pyplot as plt

i = 0

input_img = x_train[i]
output = yolo.predict(np.expand_dims(input_img,0))
# output = yolo.predict(x_test[0:2],0)

#
from yolo_utils import yoloBoxes, reshapeYoloBoxes, _nms

y1 = yoloBoxes(output[0], anchors[0], num_classes)
y2 = yoloBoxes(output[1], anchors[1], num_classes)
y3 = yoloBoxes(output[2], anchors[2], num_classes)
outputs=(y1,y2,y3)

#

boxes, scores, classes, valid_detections = _nms(outputs, \
     num_classes, \
     max_output_size=100, \
     iou_threshold=0.5, \
     score_threshold=0.5, \
     soft_nms_sigma=0.)
    
vd = valid_detections[0]
# print(boxes)
#%%
from draw_functions import draw_box

img = input_img.copy()
for box in boxes[0,:vd]:
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    w = (x2 - x1)
    h = (y2 - y1)
    x = x1 + w/2
    y = y1 + h/2
    img = draw_box(img, [x,y,w,h])
plt.imshow(img)
print(f'Detections: {num_detections}')
