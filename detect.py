import config

img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.EPOCHS
batch_size=config.BATCH_SIZE
learning_rate = config.LEARNING_RATE # TODO

#%% Model
import matplotlib.pyplot as plt
from model import build_model
# from weight_converter_model import build_model
import tensorflow as tf

# # VOC2012
# num_classes = 20
# Sound
num_classes = 4
#
yolo = build_model(num_classes)

# yolo.load_weights('weights/yolov3_transferlearning_1.h5')
# yolo.load_weights('weights/yolov3_voc2012.h5')
# yolo.load_weights('checkpoints/yolov3_voc2012_github_loss5.tf')
# yolo.load_weights('checkpoints/yolov3_voc2012_github_loss2.tf')
# yolo.load_weights('checkpoints/yolov3_voc2012_my_loss_pre3.tf')

# yolo.load_weights('checkpoints/yolov3_sound_my_loss4.tf')
# yolo.load_weights('checkpoints/yolov3_sound_from_scratch16.tf')
yolo.load_weights('checkpoints/yolov3_scratch_noiseAnchors_16.tf')

# =============================================================================
# load data
# =============================================================================

# =============================================================================
# data_dir = '../yolov3_0/data_animals' # TODO move data in separate folder
# 
# from yolo_utils import (
#     loadDictoniaries, 
#     loadImagesAndTargets, 
#     convertToTargetFormat,
#     reshapeImages
#     )
# 
# images_dict, labels_dict = loadDictoniaries(data_dir)
# 
# x, y = loadImagesAndTargets(images_dict, labels_dict)
# 
# # Prepare Images
# # Reshape
# x_reshaped = reshapeImages(x,img_size)          
# 
# # Convert y in pre-target shape. Pre-target shape is needed for split
# # Format: targets[image][grid_scale][grid,grid,(c,x,y,w,h,(classes))*anchors]
# y_targets = convertToTargetFormat(y, anchors, grid_list, num_classes)
# 
# # Split into training and test set
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x_reshaped, 
#                                                     y_targets, 
#                                                     random_state=0)
# =============================================================================

# Load voc2012 dataset
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
import yolov3_tf2.dataset as ds
import numpy as np
import tensorflow as tf

#%%
tfrecord = './data/sound_val.tfrecord'
classes = './data/sound.names'
size = 416
# anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#                          (59, 119), (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 416
anchors = config.ANCHORS
anchors = np.array(anchors[1]+anchors[1]+anchors[0], dtype=np.float32)
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

dataset = ds.load_tfrecord_dataset(
            tfrecord, classes, size)
dataset = dataset.shuffle(512)
#
import time
start = time.time()
img_raw, _label = next(iter(dataset.take(1)))

check0 = time.time()
print(f'Picking image: {check0-start}s')

img = tf.expand_dims(img_raw, 0)
img = transform_images(img, size)

_label = tf.expand_dims(_label, 0)

# Only for testing
targets = ds.transform_targets(_label, anchors, anchor_masks, size)

import numpy as np

input_img = np.array(img)
check1 = time.time()
print(f'Preparing image for input: {check1-check0}s')

output = yolo(input_img)
check2 = time.time()
print(f'Prediction and raw output: {check2-check1}s')
#
from yolo_utils import yoloBoxes, _nms
anchors = config.ANCHORS
y1 = yoloBoxes(output[0], anchors[0], num_classes)
y2 = yoloBoxes(output[1], anchors[1], num_classes)
y3 = yoloBoxes(output[2], anchors[2], num_classes)
outputs=(y1,y2,y3)
check3 = time.time()
print(f'Decoding output: {check3-check2}s')

boxes, scores, classes, valid_detections = _nms(outputs, \
     num_classes, \
     max_output_size=100, \
     iou_threshold=0.4, \
     score_threshold=0.5, \
     soft_nms_sigma=0.)
    
check4 = time.time()
print(f'nms: {check4-check3}s')
vd = valid_detections[0]
#
from draw_functions import draw_box, draw_output

# img = x_test[i].copy()
img = input_img[0].copy()
plt.figure()

img = draw_output(img, boxes, scores, classes, valid_detections)
plt.imshow(img)

print(boxes[0,:vd])
print(scores[0,:vd])
# print(scores)
print(classes)
print(f'Detections: {vd}')
end = time.time()
print(f'Draw boxes: {end-check4}s\nTotal: {end-start}s')
# import cv2
# cv2.imwrite('output.png', img)

# #%%

# # Output of detect.py
# import numpy as np
# for i, x in enumerate(output):
#     np.save(f'test_outputs/train_out_{i}', np.array(output[i]))

# # voc2012 dataset
# for i, x in enumerate(output):
#     np.save(f'test_outputs/train_true_{i}', np.array(targets[i]))
