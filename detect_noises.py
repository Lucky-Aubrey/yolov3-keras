'''
This script detects images in the folder detection_input and
stores the results in detection_output as csv-files.
'''


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

#
yolo = build_model(num_classes)

yolo.load_weights('checkpoints/model_darknet_original.tf') # same as below
# yolo.load_weights('checkpoints/yolov3_sound_my_loss4.tf')
# yolo.load_weights('checkpoints/yolov3_sound_from_scratch16.tf')
# yolo.load_weights('checkpoints/yolov3_scratch_noiseAnchors_16.tf')
# Load voc2012 dataset

import numpy as np

#%%
size = 416
anchors = config.ANCHORS
anchors = np.array(anchors[1]+anchors[1]+anchors[0], dtype=np.float32)
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

import numpy as np
import cv2
from yolo_utils import yoloBoxes, _nms
import glob
import time

im_folder = 'detection_input'
pred_folder = 'detection_output/'
frequency_range = [1, 16] # khz
duration = 10.5 # s

name_list = glob.glob(im_folder + '/' + '*.png')

seconds = []
for name in name_list:
    name = name[len(im_folder)+1:]
    input_img = plt.imread(im_folder + '/' +name )[...,:3]
    input_img = input_img / np.max(input_img)
    img = cv2.resize(input_img, (416,416), interpolation = cv2.INTER_AREA)
    img = img
    img = np.expand_dims(img, axis = 0)
    
    t_s = time.time()
    output = yolo(img)
    #
    anchors = config.ANCHORS
    y1 = yoloBoxes(output[0], anchors[0], num_classes)
    y2 = yoloBoxes(output[1], anchors[1], num_classes)
    y3 = yoloBoxes(output[2], anchors[2], num_classes)
    outputs=(y1,y2,y3)
    
    boxes, scores, classes, valid_detections = _nms(outputs, \
          num_classes, \
          max_output_size=100, \
          iou_threshold=0.5, \
          score_threshold=0.5, \
          soft_nms_sigma=0.)
        
    t_e = time.time()
    print(t_e-t_s)
    vd = valid_detections[0]
    
    boxes = boxes[0].numpy()
    scores = scores.numpy()
    classes = classes[0].numpy()
    
    res = []
    for b, p in zip(boxes, classes):
        x1,y1,x2,y2 = b
        start = x1 * duration
        end = x2 * duration
        # invert because flipping coordinate system
        a = 1 - y1
        y1 = 1 - y2
        y2 = a
        
        fr_l = (1/15 + y1) *15
        fr_h = (1/15 + y2) *15
        label = [p,start,end,fr_l,fr_h]
        res.append(label)
    
    res = np.array(res)
    seconds.append(t_e-t_s)
    
    # np.savetxt(f'{pred_folder}{name[:-4]}.csv', res , delimiter=",")
    
seconds = np.array(seconds)
np.savetxt('speed.csv', seconds , delimiter=",")
