# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:25:47 2022

@author: Lucky
"""

import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pathlib
from tensorflow import math
import tensorflow as tf

def loadDictoniaries(data_dir):
    data_dir = pathlib.Path(data_dir)

    images_dict = {
        'buffalo': list(sorted(data_dir.glob('buffalo/*.[j,J]*'))),
        'elephant': list(sorted(data_dir.glob('elephant/*.[j,J]*'))),
        'rhino': list(sorted(data_dir.glob('rhino/*.[j,J]*'))),
        'zebra': list(sorted(data_dir.glob('zebra/*.[j,J]*'))),
        }    

    labels_dict = {
        'buffalo': list(sorted(data_dir.glob('buffalo/*.txt'))),
        'elephant': list(sorted(data_dir.glob('elephant/*.txt'))),
        'rhino': list(sorted(data_dir.glob('rhino/*.txt'))),
        'zebra': list(sorted(data_dir.glob('zebra/*.txt'))),
        }
    
    return images_dict, labels_dict

def loadImagesAndTargets(images_dict, labels_dict):
    x_images, y_label = [], []
    
    # Load images
    for name, images in images_dict.items():
        for image in images:
            img = cv2.imread(str(image))
            x_images.append(img)
       
    # Load headers for images
    for name, labels in labels_dict.items():
        for label in labels:
            df = pd.read_csv(str(label), delimiter=' ', header=None)
            lbl = df.to_numpy()
            y_label.append(lbl)
    
    return x_images, y_label

# Gets iou of two boxes, assuming the have same center point
# target: (class, x,y,w,h)
def anchor_iou(anchor,target):
    anchor_box = [0.,0.,anchor[0],anchor[1]]
    target_box = [0.,0.,target[3],target[4]]
    return intersectionOverUnion(anchor_box, target_box)

# Get the best anchors for a target using the iou in descending order
# TODO source for this idea
def best_anchors(target,anchors):
    # Get iou for target with every anchor
    iou_list = []
    for g_id, grid in enumerate(anchors):
        for a_id, anchor in enumerate(grid):
            iou = anchor_iou(anchor, target)
            iou_list.append({'iou':iou, 'grid_id': g_id, 'anchor_id': a_id})
    # Sort by iou (descending)
    def sortIou(e):
        return e['iou']
    iou_list.sort(key=sortIou, reverse=True)
    # keep anchor and grid ids of best iou and all with ious > threshold
    for i in range(1,len(iou_list)):
        if iou_list[i]['iou'] < 0.5:
            del iou_list[i:]
            break
    
    return iou_list
# TODO rework with tensorflow functions
# shape of grid format: [i_label, grid_y, grid_x, anchor, (5+classes)]
def convertToTargetFormat(y_unconverted, anchors, grid_list, num_classes):
    y_converted = []
    for instance in y_unconverted:
        instance_target = []
        for g_id, g in enumerate(grid_list):
            amount_anchors = len(anchors[g_id])
            cells = np.zeros((g,g,amount_anchors,(5+num_classes)))
            instance_target.append(cells)
        for box in instance:
            anchor_ids = best_anchors(box, anchors)
            for ids in anchor_ids:
                grid_id = ids['grid_id']
                grid = grid_list[grid_id]
                anchor_id = ids['anchor_id']
                col = int(box[2]*grid)
                row = int(box[1]*grid)
                if instance_target[grid_id][col,row,anchor_id,0] != 0:
                    continue # TODO test
                x = box[1]
                y = box[2]
                w = box[3]
                h = box[4]
                classes = [0]*num_classes
                classes[int(box[0])] = 1 # example: [0, 0, 1, 0] -> class: 2
                instance_target[grid_id][col,row,anchor_id] = \
                    [1, x, y, w, h]+classes
        y_converted.append(instance_target)
    return y_converted
    
def reshapeImages(x, img_size):
    x_reshaped = []
    for img in x:
        img = cv2.resize(img,(img_size,img_size))
        x_reshaped.append(img)
    return np.array(x_reshaped).astype("float32") / 255
        
# TODO delete
# =============================================================================
# def loadDataFromDictionaries(img_dict, lbl_dict, img_size, grid, num_classes):
#     x_images, y_label = [], []
#     
#     # Load images
#     for name, images in img_dict.items():
#         for image in images:
#             img = cv2.imread(str(image))
#             resized_img = cv2.resize(img,(img_size,img_size))
#             x_images.append(resized_img)
#        
#     # Load headers for images
#     for name, labels in lbl_dict.items():
#         for label in labels:
#             df = pd.read_csv(str(label), delimiter=' ', header=None)
#             lbl = df.to_numpy()
#             cell = np.zeros((grid,grid,5+num_classes))
#     
#             # =====================================================================
#             # Save labels for each grid cell in a list. Cells are indexed starting
#             # from top left going in x direction and then in y direction, 
#             # when reaching last column. labels: [obj, x, y, w, h, classes]
#             # on grid scale!!!
#             # =====================================================================
#     
#             for box in lbl:
#                 # Scale to grid [i_laber, grid_y, grid_x, anchor, (5+classes)]
#                 col = int(box[2]*grid)
#                 row = int(box[1]*grid)
#                 x = box[1]*grid-col
#                 y = box[2]*grid-row
#                 w = box[3]*grid
#                 h = box[4]*grid
#                 classes = [0]*num_classes
#                 classes[int(box[0])] = 1 # example: [0, 0, 1, 0] -> class: 2
#                 cell[col][row]=[1, x, y, w, h]+classes
#                 
#             # y_label: (index,grid_y,grid_x,(obj, x, y, w, h, classes))
#             y_label.append(cell)
#     
#     # Convert to arrays
#     x_images = np.array(x_images)
#     y_label = np.array(y_label)
#     print(x_images.shape)            
# 
#     x_train, x_test, y_train, y_test = train_test_split(x_images, 
#                                                         y_label, 
#                                                         random_state=0)
#     x_train = x_train.astype("float32") / 255
#     x_test = x_test.astype("float32") / 255
#     return x_train, x_test, y_train, y_test
# =============================================================================

# converts [BS][GRID][GRID][x,y,w,h] to [BS][GRID][GRID][xmin,ymin,xmax,ymax]
# BS: Batch size, GRID: grid cell per row/column
def box_xywh_to_sides(box): # TODO change name to tf...
    xmin = box[...,0]-box[...,2]/2
    xmax = box[...,0]+box[...,2]/2
    ymin = box[...,1]-box[...,3]/2
    ymax = box[...,1]+box[...,3]/2
    return tf.stack([xmin,ymin,xmax,ymax],axis=-1)
    
# Input for boxes: [BS][GRID][GRID][x,y,w,h]
def intersection_over_union(boxA,boxB, xywh=True): # TODO change name to tf...
    if xywh == True:
        # shape: [BS][GRID][GRID][xmin,xmax,ymin,ymax]
        boxA = box_xywh_to_sides(boxA)
        boxB = box_xywh_to_sides(boxB)
        
	# determine the (x, y)-coordinates of the intersection rectangle
    ximin = math.maximum(boxA[...,0], boxB[...,0])
    yimin = math.maximum(boxA[...,1], boxB[...,1])
    ximax = math.minimum(boxA[...,2], boxB[...,2])
    yimax = math.minimum(boxA[...,3], boxB[...,3])
    #compute the area of intersection rectangle
    interArea = math.maximum(0., ximax - ximin)*math.maximum(0., yimax - yimin)
    #compute the area of both the prediction and ground-truth
    #rectangles
    boxAArea = (boxA[...,2] - boxA[...,0]) * (boxA[...,3] - boxA[...,1])
    boxBArea = (boxB[...,2] - boxB[...,0]) * (boxB[...,3] - boxB[...,1])
    # Compute the intersection over union by taking the intersection
    # Area and dividing it by the sum of prediction + ground-truth
    # Areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
 
    return iou

def broadcast_iou(box_1,box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))
    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    
    return intersection_over_union(box_1, box_2, xywh=False)

# TODO delete
# def _meshgrid(n_a, n_b):

#     return [
#         tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a)),
#         tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a))
#     ]

def _meshgrid(n_a, n_b):

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]

def yoloBoxes(output, anchors, num_classes):
    grid_size = tf.shape(output)[1:3]
    
    bboxes_xy, bboxes_wh, conf, p_class = tf.split(
        output, (2,2,1,num_classes), axis = -1)
    
    bboxes_xy = tf.sigmoid(bboxes_xy)    
    conf = tf.sigmoid(conf)
    p_class = tf.sigmoid(p_class)
    
    grid = _meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    
    bboxes_xy = (bboxes_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    bboxes_wh = tf.exp(bboxes_wh) * anchors
    
    bboxes_x1y1 = bboxes_xy - bboxes_wh / 2
    bboxes_x2y2 = bboxes_xy + bboxes_wh / 2
    
    bboxes = tf.concat([bboxes_x1y1, bboxes_x2y2], -1)
    # bboxes = tf.concat([bboxes_xy, bboxes_wh], -1)

    return bboxes, conf, p_class

def reshapeYoloBoxes(outputs):
    b,c,p = [],[],[]
    # for o in outputs:
    #    b.append(tf.reshape(o[0], (-1, tf.shape(o[0])[-1])))
    #    c.append(tf.reshape(o[1], (-1, tf.shape(o[1])[-1])))
    #    p.append(tf.reshape(o[2], (-1, tf.shape(o[2])[-1])))
    # TODO check if above is sufficient
    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        p.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    
    b = tf.concat(b,1)
    c = tf.concat(c,1)
    p = tf.concat(p,1)
    
    return b,c,p

def _nms(outputs, num_classes, max_output_size, iou_threshold, score_threshold, \
         soft_nms_sigma):
    bbox,conf,prob_class = reshapeYoloBoxes(outputs)
    
    if num_classes == 1:
        scores = conf
    else:
        scores = conf * prob_class
        
    # classes = tf.argmax(scores,1)

    # scores = tf.reduce_max(scores,[1])
    
    if tf.shape(bbox)[0]>1:
        res = []
        for iscores, ibbox in zip(scores,bbox):
            idscores = iscores
            iscores = tf.reduce_max(idscores,[1])
            ibbox = tf.reshape(ibbox,(-1,4))
            iclasses = tf.argmax(idscores,1)
            
            iselected_indices, iselected_scores = tf.image.non_max_suppression_with_scores(
                    boxes=ibbox,
                    scores=iscores,
                    max_output_size=max_output_size,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                    soft_nms_sigma=soft_nms_sigma
                )
            inum_valid_nms_boxes = tf.shape(iselected_indices)[0]
            
            iboxes=tf.gather(ibbox, iselected_indices)
            iboxes = tf.expand_dims(iboxes, axis=0)
            iscores=iselected_scores
            iscores = tf.expand_dims(iscores, axis=0)
            iclasses = tf.gather(iclasses,iselected_indices)
            iclasses = tf.expand_dims(iclasses, axis=0)
            ivalid_detections=inum_valid_nms_boxes
            ivalid_detections = tf.expand_dims(ivalid_detections, axis=0)
            
            res.append((iboxes, iscores, iclasses, ivalid_detections))
        
        return res

            
            
    dscores = tf.squeeze(scores, axis=0) # TODO enable nms for multiple images
    scores = tf.reduce_max(dscores,[1])
    bbox = tf.reshape(bbox,(-1,4))
    classes = tf.argmax(dscores,1)

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            boxes=bbox,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            soft_nms_sigma=soft_nms_sigma
        )
    
    num_valid_nms_boxes = tf.shape(selected_indices)[0]
    
    boxes=tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)
    scores=selected_scores
    scores = tf.expand_dims(scores, axis=0)
    classes = tf.gather(classes,selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    valid_detections=num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)
    # print(selected_indices)
    # print(boxes)
    # print(scores)
    # print(classes)
    # print(valid_detections)
    return boxes, scores, classes, valid_detections

# TODO maybe use tf.image.non_max_suppresion
def nonMaximumSupression(bboxes, thresh_iou, thresh_conf):
    # bboxes = [grid*grid, (class, conf, x, y, w, h)]
    
    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > thresh_conf]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    # bboxes_after_nms = []
    
    i=0
    while i <= len(bboxes):
        j = i +1
        while j < len(bboxes):
            iou = intersectionOverUnion(bboxes[i][2:], bboxes[j][2:])
            if iou > thresh_iou:
                del bboxes[j]
            else:
                j +=1
        i += 1
                    
        
    
    # for i, box1 in enumerate(bboxes):
    #     bboxes_rm = bboxes.copy()
    #     del bboxes[i]
    #     for box2 in bboxes_rm:
    #         iou = intersection_over_union(box1[1:5], box2[1:5])
    #         if iou < thresh_iou: # and box1[0] == box2[0] # TODO same class
            
    return bboxes

# Numpy version
def xywhToSides(box):
    xmin = box[0]-box[2]/2
    xmax = box[0]+box[2]/2
    ymin = box[1]-box[3]/2
    ymax = box[1]+box[3]/2
    return [xmin,xmax,ymin,ymax]
# Numpy version
def intersectionOverUnion(boxA,boxB):
    # boxA = xywhToSides(boxA)
    # boxB = xywhToSides(boxB)
    
    # shape: [BS][GRID][GRID][xmin,xmax,ymin,ymax]
    
	# determine the (x, y)-coordinates of the intersection rectangle
    ximin = np.maximum(boxA[0], boxB[0])
    yimin = np.maximum(boxA[1], boxB[1])
    ximax = np.minimum(boxA[2], boxB[2])
    yimax = np.minimum(boxA[3], boxB[3])

    #compute the area of intersection rectangle
    interArea = np.maximum(0., ximax - ximin)*np.maximum(0., yimax - yimin)
    
    #compute the area of both the prediction and ground-truth
    #rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the intersection over union by taking the intersection
    # Area and dividing it by the sum of prediction + ground-truth
    # Areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    return iou
 
# Loading dataset
import os
def get_label(file_path):
    return tf.strings.split(file_path, os.path.sep)[-2]

def scale(img, label):
    return img/255, label

def process_image_for_classification(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img)[...,:3]
    img = tf.image.resize(img, [416,416])
    return img, label

table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(['artefact', 'click', 'squeal', 'wirebrush']),
        values=tf.constant([0, 1, 2, 3]),
    ),
    default_value=tf.constant(-1),
    name="class_weight"
)
def detection_label(file_path):
    class_label = tf.strings.split(file_path, os.path.sep)[-2]
    
    if class_label != 'no_sound':
        file_name = tf.strings.split(file_path, os.path.sep)[-1]
        chopped_length = tf.strings.length(file_name)-4
        file_name = tf.strings.substr(file_name, 0, chopped_length)
        
        folder = 'data/sound_evaluation/labels_numpy/'
        file_name = folder + file_name.numpy() +'.npy'
        print(file_name)
        file_name = np.load(file_name)
    else:
        file_name = ''
        # TODO dosomething
    return class_label, file_name

def process_image_for_detection(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img)[...,:3]
    img = tf.image.resize(img, [416,416])
    
    label = detection_label(file_path)
    
    return img, label
