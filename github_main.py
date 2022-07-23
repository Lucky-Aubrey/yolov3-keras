
import config

# file_name = 'yolov3_scratch_1'
img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.EPOCHS
batch_size=config.BATCH_SIZE
learning_rate = config.LEARNING_RATE # TODO

#%% Test github_load_dataset voc2012
from github_load_datasets import get_voc2012_datasets
import numpy as np
train_dataset, val_dataset = get_voc2012_datasets()

num_classes = 20
anchors = yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
batch_size = 16


#%% Model

from github_model import (
YoloV3, YoloLoss
    )
import tensorflow as tf
# from model import build_model_2
# from model_github import YoloV3
# yolo = YoloV3(size=416, training=True)
# Create model for <num_classes>
yolo = YoloV3(size = 416, training=True, classes = num_classes) 

# Create model and load it with weights of a pretrained coco model (80 classes)
coco_yolo = YoloV3(size = 416, training=True, classes = 80)
coco_yolo.load_weights('yolov3_coco.tf')

# Get weights of the darknet layers and set it into the corresponding darknet
# layers
yolo.get_layer('yolo_darknet').set_weights(
    coco_yolo.get_layer('yolo_darknet').get_weights())
# # freeze darknet
# def freeze_all(model, frozen=True):
#     model.trainable = not frozen
#     if isinstance(model, tf.keras.Model):
#         for l in model.layers:
#             freeze_all(l, frozen)
# def unfreeze_all(model, frozen=False):
#     freeze_all(model, frozen)
# freeze_all(yolo.get_layer('yolo_darknet')) # TODO uncomment
# Conv0
# yolo.get_layer('yolo_conv_0').set_weights(
#     coco_yolo.get_layer('yolo_conv_0').get_weights())
# # Conv1
# yolo.get_layer('yolo_conv_1').set_weights(
#     coco_yolo.get_layer('yolo_conv_1').get_weights())
# # Conv2
# yolo.get_layer('yolo_conv_2').set_weights(
#     coco_yolo.get_layer('yolo_conv_2').get_weights())

# # Build loss function for all 3 outputs
# loss_function = [yoloLoss(anchors[0], weights, num_classes),
#                  yoloLoss(anchors[1], weights, num_classes),
#                  yoloLoss(anchors[2], weights, num_classes)]
# Build loss function for all 3 outputs
loss_function = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]
# Compile, set optimizers and set lossfunction
yolo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate),
             loss=loss_function)

# Add callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
callbacks = [ReduceLROnPlateau(verbose=1), EarlyStopping(patience=3, verbose=1)]

# =============================================================================
# # Train model
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

# Train model with voc2012
# history = yolo.fit(x_train, y_train, batch_size=batch_size, epochs=10,
#                     # validation_data = [x_train, y_train]
#                     # validation_data = [x_val, y_val]
#                     validation_data=(x_test, y_test),
#                     callbacks = [callbacks]
#                     ) 
# # Unfreeze darknet
# unfreeze_all(yolo.get_layer('yolo_darknet'))
history = yolo.fit(train_dataset, batch_size=batch_size, epochs=epochs,
                    # validation_data = [x_train, y_train]
                    # validation_data = [x_val, y_val]
                    validation_data=val_dataset,
                    callbacks = [callbacks]
                    ) 

file_name = 'github_yolov3'
yolo.save_weights('weights/'+file_name+'.h5')
np.save('history/'+file_name+'.npy', history.history)