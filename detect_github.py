import config

file_name = 'yolov3_2'
img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.EPOCHS
batch_size=config.BATCH_SIZE
learning_rate = config.LEARNING_RATE # TODO

# =============================================================================
# load data
# =============================================================================
data_dir = '../yolov3_0/data_animals' # TODO move data in separate folder

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

#%% Model
import matplotlib.pyplot as plt
# from model import build_model
from model_github import YoloV3
import tensorflow as tf

# COCO
num_classes = 80
#
yolo = YoloV3(yoloBoxOutputs=True)

yolo.load_weights(f'weights/yolov3_coco_github')

#%%
import numpy as np

# i = 500
# i+=1
# i-=1

input_img = x_reshaped[i]
outputGit = yolo.predict(np.expand_dims(input_img,0))

y1=tf.concat([outputGit[0][0], outputGit[0][1], outputGit[0][2]],-1)
y2=tf.concat([outputGit[1][0], outputGit[1][1], outputGit[1][2]],-1)
y3=tf.concat([outputGit[2][0], outputGit[2][1], outputGit[2][2]],-1)
outputGityolo = (y1,y2,y3)

boxes, scores, classes, valid_detections = yolo.predict(
    np.expand_dims(input_img,0))

v_d = valid_detections[0]
#

from draw_functions import draw_box

# img = x_test[i].copy()
img = input_img.copy()
for box in boxes[0,:v_d]:
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
print(boxes[0,:v_d])
print(scores[0,:v_d])
print(classes[0,:v_d])
print(f'Detections: {valid_detections}')

# import cv2
# cv2.imwrite('output.png', img)
