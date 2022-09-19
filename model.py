# -*- coding: utf-8 -*-
"""
Created on Thu May 12 17:18:24 2022

@author: Lucky
"""

import keras
from keras import Input
from keras.layers import (
    BatchNormalization, Conv2D, UpSampling2D, Concatenate, ZeroPadding2D, Lambda
    )
from keras.layers import add, LeakyReLU
import tensorflow as tf

import numpy as np
from tensorflow import math

# Code is based of https://towardsdatascience.com/yolo-v3-object-detection-with-keras-461d2cfccef6
#
# Format for the arguments of conv_block (example)
# x: input, filter_number: 32, kernel_size: 3, stride: <empty> or 2

def conv_block(x, filter_number, kernel_size, stride=1, batch_norm=True):
    # padding = 'same'
    if stride == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filter_number, 
               kernel_size=(kernel_size,kernel_size), 
               strides=(stride,stride), 
               padding = padding, 
               use_bias=not batch_norm)(x)
    if batch_norm is True:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x

# Format for the arguments of res_block (example)
# x: input, filter_list: [32,64], repeats: 2

def res_block(x, filter_list, repeats):
    for i in range(repeats):
        res_skip = x
        x = conv_block(x, filter_list[0], 1)
        x = conv_block(x, filter_list[1], 3)
        x = add([res_skip, x])
    return x

# Format for the arguments of filters_numbers, kernel_sizes, strides
# [32,64,...], [1,3,...], [1,2,...] <- all same length
def conv_block_2(x, filter_numbers, kernel_sizes, strides):
    
    for i, filter_number in enumerate(filter_numbers):
        kernel_size = kernel_sizes[i]
        stride = strides[i]
        x = conv_block(x, filter_number, kernel_size, stride)
    return x


def darknet(name):
    
    x = inputs = Input([None, None, 3])
    x = conv_block(inputs, 32, 3)
    x = conv_block(x, 64, 3, 2)
    x = res_block(x, [32, 64], 1)
    
    x = conv_block(x, 128, 3, 2)
    x = res_block(x, [64, 128], 2)
    
    x = conv_block(x, 256, 3, 2)
    x = x_36 = res_block(x, [128, 256], 8)
    
    x = conv_block(x, 512, 3, 2)
    x = x_61 = res_block(x, [256, 512], 8)
    
    x = conv_block(x, 1024, 3, 2)
    x = res_block(x, [512, 1024], 4)
    return tf.keras.Model(inputs, [x_36, x_61, x], name=name)

def yolo_conv(x_in, filters, kernels, strides, filters_up, name):
    if isinstance(x_in, tuple):
        inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
        x, x_skip = inputs

        # concat with skip connection
        x = conv_block(x, filters_up, 1, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, x_skip])
    else:
        x = inputs = Input(x_in.shape[1:])
    
    x = conv_block_2(x, filters, kernels, strides)
    return tf.keras.Model(inputs, x, name=name)(x_in)

def yolo_output(x_in, filters, anchors, num_classes, name):
    x = inputs = Input(x_in.shape[1:])
    x = conv_block(x, filters, 3, 1)
    x = conv_block(x, (5+num_classes)*3, 1, 1, False)
    x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, num_classes + 5)))(x)
    return tf.keras.Model(inputs, x, name=name)(x_in)


def build_model(num_classes, shape=(416, 416, 3)):
    
    x = inputs = Input(shape=shape, name='input')
    
    #Darknet53
    x_36, x_61, x = darknet('yolo_darknet')(x)
    
    # 13 by 13 detection head
    x = yolo_conv(x, [512, 1024, 512, 1024, 512, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1],
                      None,
                      name='yolo_conv_0')
    
    out0 = yolo_output(x, 1024, 3, num_classes, name='yolo_output_0')
    
    # 26 by 26 detection head
    x = yolo_conv((x, x_61), [256, 512, 256, 512, 256, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1],
                      256,
                      name='yolo_conv_1')
    
    out1 = yolo_output(x, 512, 3, num_classes, name='yolo_output_1')

    
    # 52 by 52 detection head
    x = yolo_conv((x,x_36), [128, 256, 128, 256, 128, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1],
                      128,
                      name='yolo_conv_2')
    
    out2 = yolo_output(x, 256, 3, num_classes, name='yolo_output_2')
    
    outputs = [out0, out1, out2]
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='darknet_53')

# TODO delete
def build_model_2(num_classes, shape=(416, 416, 3)):
    
    #Darknet53
    inputs = Input(shape=shape)
    
    x = conv_block(inputs, 32, 3)
    x = conv_block(x, 64, 3, 2)
    x = res_block(x, [32, 64], 1)
    
    x = conv_block(x, 128, 3, 2)
    x = res_block(x, [64, 128], 2)
    
    x = conv_block(x, 256, 3, 2)
    x = x_36 = res_block(x, [128, 256], 8)
    
    x = conv_block(x, 512, 3, 2)
    x = x_61 = res_block(x, [256, 512], 8)
    
    x = conv_block(x, 1024, 3, 2)
    x = res_block(x, [512, 1024], 4)
    
    # End Darknet53
    # 13 by 13 detection head
    x = skip1 = conv_block_2(x, [512, 1024, 512, 1024, 512, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1])
    x = conv_block(x, 1024, 3, 1)
    x = conv_block(x, (5+num_classes)*3, 1, 1, False)
    
    # 26 by 26 detection head
    skip1 = conv_block(skip1, 256, 1, 1)
    skip1 = UpSampling2D(2)(skip1)
    x1 = Concatenate()([x_61,skip1])
    x1 = skip2 = conv_block_2(x1, [256, 512, 256, 512, 256, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1])
    x1 = conv_block(x1, 512, 3, 1)
    x1 = conv_block(x1, (5+num_classes)*3, 1, 1, False)
    
    # 52 by 52 detection head
    skip2 = conv_block(skip2, 128, 1, 1)
    skip2 = UpSampling2D(2)(skip2)
    x2 = Concatenate()([x_36,skip2])
    x2 = conv_block_2(x2, [128, 256, 128, 256, 128, ], 
                      [1,3,1,3,1], 
                      [1,1,1,1,1])
    x2 = conv_block(x2, 256, 3, 1)
    x2 = conv_block(x2, (5+num_classes)*3, 1, 1, False)
    
    outputs = [x, x1, x2]
    return keras.Model(inputs=inputs, outputs=outputs, name='darknet_53')


from yolo_utils import _meshgrid, broadcast_iou
from tensorflow.keras.losses import (
    BinaryCrossentropy,
    CategoricalCrossentropy,
    SparseCategoricalCrossentropy,
    # sparse_categorical_crossentropy # bce used instead
)


bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


    
def yoloLoss(anchors, weights, num_classes):
    def Lossfunction(y_true, y_pred):
        # Already done in yolo_out
        # # Reshape output of model to [BS,GRID,GRID,3,5+num_classes]
        # y_pred = tf.reshape(y_pred, \
        #                     (-1,y_pred.shape[1],y_pred.shape[2],
        #                       3,5+num_classes))
        
        # Create meshgrids for transformation from rel. to abs. coord. and vice versa
        grid_size = tf.shape(y_true)[1:3]
        grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2) 
        
# =============================================================================
#         # TODO change to front
#         true_obj = y_true[...,0]
#         true_obj = tf.expand_dims(true_obj,-1)
#         abs_true_xy = y_true[...,1:3]
#         abs_true_wh = y_true[...,3:5]
#         true_class = y_true[...,5] # TODO label encoding
#         
#         pred_obj = tf.sigmoid(y_pred[...,0])
#         pred_obj = tf.expand_dims(pred_obj,-1)
#         rel_pred_xy = y_pred[...,1:3]
#         rel_pred_wh = y_pred[...,3:5]
#         pred_class = tf.sigmoid(y_pred[...,5:])
# =============================================================================
        
        # TODO delete
        # conf at [4]
        true_obj = y_true[...,4]
        true_obj = tf.expand_dims(true_obj,-1)
        abs_true_xy = (y_true[...,0:2] + y_true[...,2:4])/2
        abs_true_wh = y_true[...,2:4] - y_true[...,0:2]
        true_class = y_true[...,5] # TODO label encoding
        
        pred_obj = tf.sigmoid(y_pred[...,4])
        pred_obj = tf.expand_dims(pred_obj,-1)
        rel_pred_xy = y_pred[...,0:2]
        rel_pred_wh = y_pred[...,2:4]
        pred_class = tf.sigmoid(y_pred[...,5:])
        
        
        # Now specific coordinates for the losses are needed:
        # xy_loss
        sig_pred_xy = tf.sigmoid(rel_pred_xy)
        sig_true_xy = (abs_true_xy * tf.cast(grid_size, tf.float32) - \
                        tf.cast(grid, tf.float32)) # sigma(rel_pred_xy)
        
        box_loss_scale = 2 - abs_true_wh[...,0] * abs_true_wh[...,1]
        
        # wh_loss
        rel_pred_wh = tf.math.minimum(tf.where(tf.math.is_inf(rel_pred_wh), 
                               tf.zeros_like(rel_pred_wh), rel_pred_wh), 88.)
        rel_true_wh = tf.math.log(abs_true_wh/anchors)
        rel_true_wh = tf.where(tf.math.is_inf(rel_true_wh), 
                               tf.zeros_like(rel_true_wh), rel_true_wh)

        # coordinates for broadcast_iou
        abs_true_box = y_true[...,:4]
        abs_pred_xy = (sig_pred_xy + tf.cast(grid, tf.float32)) /  \
            tf.cast(grid_size, tf.float32)
            
        # clip for high values to avoid inf
        abs_pred_wh = tf.math.exp(rel_pred_wh)*anchors
        abs_pred_x1y1 = abs_pred_xy - abs_pred_wh/2
        abs_pred_x2y2 = abs_pred_xy + abs_pred_wh/2
        abs_pred_box = tf.concat([abs_pred_x1y1, abs_pred_x2y2], -1)
        
        # obj_loss
        obj_mask = tf.squeeze(true_obj, -1) #
        # construct ignore mask
        mask_true_box = tf.boolean_mask(abs_true_box, tf.cast(obj_mask, tf.bool))
        
        # input format for broadcast_iou: (x1,y1,x2,y2)
        b_iou = broadcast_iou(abs_pred_box, mask_true_box)
        best_iou = tf.reduce_max(b_iou, -1)
        ignore_thresh = 0.5
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        
        # =============================================================================
        # Losses
        # =============================================================================
        # box_loss
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(sig_true_xy - sig_pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(rel_true_wh - rel_pred_wh), axis=-1)
        
            
        # obj_loss + noobj_loss
        obj_loss_all = bce(true_obj, pred_obj)
        
        obj_loss = obj_mask * obj_loss_all
        noobj_loss = (1 - obj_mask) * ignore_mask * obj_loss_all
        
        # class_loss (blog used sparse_categorical_crossentropy
        class_loss = obj_mask * scce(
            true_class, pred_class) # TODO label encoding
        
        # sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        noobj_loss = tf.reduce_sum(noobj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        
        # only_obj_loss = tf.reduce_sum(obj_mask*bce(true_obj, pred_obj), axis=(1,2,3))
        # print(f'xy_loss: {xy_loss}')
        # print(f'wh_loss: {wh_loss}')
        # print(f'obj_loss: {only_obj_loss}')
        # print(f'noobj_loss: {obj_loss - only_obj_loss}')
        # print(f'class_loss: {class_loss}')
        
        # print('')
        # print(tf.stack([xy_loss, wh_loss, obj_loss+noobj_loss, class_loss]))


        # l_obj = tf.reduce_sum(
        #     obj_mask * bce(true_obj, pred_obj), axis=(1, 2, 3))
        # l_noobj = tf.reduce_sum((1 - obj_mask) * ignore_mask *\
        #                         bce(true_obj, pred_obj), axis=(1, 2, 3))
        
        total_loss = weights[0] * xy_loss + \
            weights[1] * wh_loss + \
            weights[2] * obj_loss +\
            weights[3] * noobj_loss +\
            weights[4] * class_loss
        return total_loss
    return Lossfunction


# =============================================================================
# Custom metrics
# =============================================================================

# =============================================================================
# Custom Callback
# =============================================================================

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, anchors, model_name, file_name):
        self.x_test = np.expand_dims(x_test, axis = 0)
        self.y_test = np.expand_dims(y_test, axis = 0)
        self.anchors = anchors
        self.model = model_name
        self.file_name = file_name
        self.predictions = []
        
    def on_train_begin(self, logs=None):
        prediction = self.model.predict(self.x_test)
        prediction = convertOutput(prediction, self.anchors)
        
        self.predictions.append(prediction)
        
    def on_train_end(self, logs=None):
        self.predictions = np.array(self.predictions)
        np.save('callbacks/cb_'+self.file_name+'.npy', self.predictions)

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.x_test)
        prediction = convertOutput(prediction, self.anchors)
        
        self.predictions.append(prediction)
        
def convertOutput(y_all, anchors):
    a_len = 9
    y_converted = []
    for inst_id, instance in enumerate(y_all):
        inst_head = []
        for g_id, scale in enumerate(instance):
            head_as = []
            for a_id, anchor in enumerate(anchors[g_id]):
                y = scale[...,a_id*a_len:(a_id+1)*a_len]
                pred_c = tf.expand_dims(math.sigmoid(y[...,0]), -1)
                pred_xy = math.sigmoid(y[...,1:3])
                pred_wh = math.exp(y[...,3:5])*anchor
                pred_class = math.sigmoid(y[...,5:])
                head_a = tf.concat([pred_c, pred_xy, pred_wh, pred_class], -1)
                head_as.append(head_a)
            
            head_as = tf.concat(head_as, -1)
            inst_head.append(head_as)
        y_converted.append(inst_head)
    return y_converted

class ModelOutput(keras.callbacks.Callback):
    def __init__(self, x_test, model_name, file_name):
        self.x_test = np.expand_dims(x_test, axis = 0)
        self.model = model_name
        self.file_name = file_name
        self.predictions = []
        
    def on_train_begin(self, logs=None):
        prediction = self.model.predict(self.x_test)
        
        self.predictions.append(prediction)
        
    def on_train_end(self, logs=None):
        # self.predictions = np.array(self.predictions)
        # np.save('callbacks/cb_'+self.file_name+'.npy', self.predictions)
        print(self.predictions[0][0].shape)
        np.save('callbacks/modelOutput13.npy', np.array(self.predictions[0][0]))
        # print(self.predictions[0].shape)

    # def on_epoch_end(self, epoch, logs=None):
    #     prediction = self.model.predict(self.x_test)
        
    #     self.predictions.append(prediction)
    
# TODO delete
# print check
def pr(x):
    print(f'\npr[:,:]:\n{x[0,:,:,2]}')
    # print(f'\npr[6,6]:\n{x[0,6,6,1]}')
def pr2(x):
    print(f'\npr2: {tf.reduce_sum(x)}')
def pr13(x):
    if tf.shape(x)[1] == 13:
        print(x[0,:13,:13,1])