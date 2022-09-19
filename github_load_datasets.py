# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 00:48:50 2022

@author: Lucky
"""

import yolov3_tf2.dataset as dataset
import config
import tensorflow as tf
import numpy as np

def get_voc2012_datasets():
    train_dataset = './data/voc2012_train.tfrecord' # TODO 
    val_dataset = './data/voc2012_val.tfrecord' # TODO 
    classes = './data/voc2012.names'
    size = 416
    batch_size = config.BATCH_SIZE
    # From repo:
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                        (59, 119), (116, 90), (156, 198), (373, 326)],
                       np.float32) / 416
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    train_dataset = dataset.load_tfrecord_dataset(
        train_dataset, classes, size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(
        val_dataset, classes, size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))
    return train_dataset, val_dataset

def get_sound_datasets():
    train_dataset = './data/sound_train.tfrecord' # TODO 
    val_dataset = './data/sound_val.tfrecord' # TODO 
    classes = './data/sound.names'
    size = config.IMG_SIZE
    batch_size = config.BATCH_SIZE
    # From repo:
    # anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
    #                     (59, 119), (116, 90), (156, 198), (373, 326)],
    #                    np.float32) / 416
    anchors = config.ANCHORS
    anchors = np.array(anchors[1]+anchors[1]+anchors[0], dtype=np.float32)
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    train_dataset = dataset.load_tfrecord_dataset(
        train_dataset, classes, size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_tfrecord_dataset(
        val_dataset, classes, size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))
    return train_dataset, val_dataset
