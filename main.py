# -*- coding: utf-8 -*-
"""
Created on Tue May 17 23:55:42 2022

@author: Lucky
"""


import config
import numpy as np

img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.EPOCHS
batch_size=config.BATCH_SIZE
learning_rate = config.LEARNING_RATE
file_name = config.file_name
mode = config.mode

print('model:', file_name)
print('mode:', mode)    
print('classes:', num_classes)

# =============================================================================
# 
# # =============================================================================
# # load data
# # =============================================================================
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
# 
# 
# # Convert into final target shape and format(numpy array)
# # Format: targets[grid_scale][image][grid,grid,(c,x,y,w,h,(classes))*anchors]
# y1,y2,y3 = list(map(list, zip(*y_train))) # TODO change in loading data
# # TODO replace with tf.tensor
# import numpy as np
# y1 = np.array(y1)
# y2 = np.array(y2)
# y3 = np.array(y3)
# y_train = [y1,y2,y3]
# 
# # Test set
# y1,y2,y3 = list(map(list, zip(*y_test))) # TODO change in loading data
# 
# y1 = np.array(y1)
# y2 = np.array(y2)
# y3 = np.array(y3)
# y_test = [y1,y2,y3]
# 
# =============================================================================
#%% Test github_load_dataset voc2012
from github_load_datasets import (
    get_voc2012_datasets,
    get_sound_datasets)

train_dataset, val_dataset = get_voc2012_datasets()
# train_dataset, val_dataset = get_sound_datasets()


#%% Model

from model import (
build_model, yoloLoss
    )
import tensorflow as tf
# from model import build_model_2
# from model_github import YoloV3
# yolo = YoloV3(size=416, training=True)
# Create model for <num_classes>
yolo = build_model(num_classes) 

# freeze darknet
def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)
def unfreeze_all(model, frozen=False):
    freeze_all(model, frozen)
    
if mode == 'darknet':
    # Create model and load it with weights of a pretrained coco model (80 classes)
    coco_yolo = build_model(80)
    coco_yolo.load_weights('yolov3_coco.tf')
    
    # Get weights of the darknet layers and set it into the corresponding darknet
    # layers
    yolo.get_layer('yolo_darknet').set_weights(
        coco_yolo.get_layer('yolo_darknet').get_weights())
elif mode == 'frozenDarknet':
    # Create model and load it with weights of a pretrained coco model (80 classes)
    coco_yolo = build_model(80)
    coco_yolo.load_weights('yolov3_coco.tf')
    
    # Get weights of the darknet layers and set it into the corresponding darknet
    # layers
    yolo.get_layer('yolo_darknet').set_weights(
        coco_yolo.get_layer('yolo_darknet').get_weights())
    # Freezing Darknet (Feature Map)
    freeze_all(yolo.get_layer('yolo_darknet')) # TODO uncomment
    
# # Conv0
# yolo.get_layer('yolo_conv_0').set_weights(
#     coco_yolo.get_layer('yolo_conv_0').get_weights())
# # Conv1
# yolo.get_layer('yolo_conv_1').set_weights(
#     coco_yolo.get_layer('yolo_conv_1').get_weights())
# # Conv2
# yolo.get_layer('yolo_conv_2').set_weights(
#     coco_yolo.get_layer('yolo_conv_2').get_weights())

# github loss function
# =============================================================================
# # TODO delete
# # Testing github_loss
# from github_model import YoloLoss
# anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#                           (59, 119), (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 416
# anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
# loss_function = [YoloLoss(anchors[mask], classes=num_classes)
#             for mask in anchor_masks]
# =============================================================================

# My loss function
# Build loss function for all 3 outputs
loss_function = [yoloLoss(anchors[0], weights, num_classes),
                  yoloLoss(anchors[1], weights, num_classes),
                  yoloLoss(anchors[2], weights, num_classes)]

# Compile, set optimizers and set lossfunction
yolo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
             loss=loss_function)

# Add callbacks
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint)
# for saving


# =============================================================================
# # Train model with animal data
# history = yolo.fit(x_train, y_train, batch_size=batch_size, epochs=10,
#                     # validation_data = [x_train, y_train]
#                     # validation_data = [x_val, y_val]
#                     validation_data=(x_test, y_test),
#                     callbacks = [callbacks]
#                     ) 
# # Unfreeze darknet
# unfreeze_all(yolo.get_layer('yolo_darknet'))
# history = yolo.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
#                     # validation_data = [x_train, y_train]
#                     # validation_data = [x_val, y_val]
#                     validation_data=(x_test, y_test),
#                     callbacks = [callbacks]
#                     ) 
# =============================================================================

# With voc2012 data

# callbacks = [ReduceLROnPlateau(verbose=1),
#              EarlyStopping(patience=3, verbose=1),
#              ModelCheckpoint('checkpoints/'+file_name+'_pre{epoch}.tf',
#                             verbose=1, save_weights_only=True),]

# # Train model with voc2012
# history = yolo.fit(train_dataset, batch_size=batch_size, epochs=5,
#                     # validation_data = [x_train, y_train]
#                     # validation_data = [x_val, y_val]
#                     validation_data=val_dataset,
#                     callbacks = [callbacks]
#                     ) 
# # Unfreeze darknet
# unfreeze_all(yolo.get_layer('yolo_darknet'))

# np.save('history/'+file_name+'_pre.npy', history.history)

callbacks = [ReduceLROnPlateau(verbose=1),
             # EarlyStopping(patience=3, verbose=1),
             ModelCheckpoint('checkpoints/'+file_name+'.tf',
                            verbose=1, save_weights_only=True,
                            save_best_only=True),]
# Train unfreezed
history = yolo.fit(train_dataset, batch_size=batch_size, epochs=epochs,
                    # validation_data = [x_train, y_train]
                    # validation_data = [x_val, y_val]
                    validation_data=val_dataset,
                    callbacks = [callbacks]
                    ) 


# file_name = 'yolov3_voc2012_github_loss'
# yolo.save_weights('weights/'+file_name+'.h5')
np.save('history/'+file_name+'.npy', history.history)