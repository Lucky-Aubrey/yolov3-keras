# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 22:17:39 2022

@author: Lucky
"""

models = {
'Fast RCNN' : ['train','VGG-16',19.7],
'Faster RCNN' : ['trainval','VGG-16',21.9],
'R-FCN' : [' trainval','VGG-16',22.6],
'CoupleNet' : ['trainval','ResNet-101',34.4],
'Faster RCNN+++' : ['trainval','ResNet-101-C4', 34.9],
'Faster RCNN w FPN' : ['trainval35k','ResNet-101-FPN', 36.2],
'Deformable R-FCN' : ['trainval','Alignmed-inception-ResNet', 37.5],
'umd-ted' : ['trainval','ResNet-101', 40.8],
'Mask RCNN' : ['trainval35k','ResNetXT-101', 39.8],
'DCNv2+Faster RCNN' : ['train118k','ResNet-101', 44.8],
'YOLOv2' : ['trainval35k','DarkNet-53', 33.0],
'YOLOv3' : ['trainval35k','DarkNet-19', 21.6],
'DSSD321' : ['trainval35k','ResNet-101', 28.0],
'SSD513' : ['trainval35k','ResNet-101', 31.2],
'DSSD513' : ['trainval35k','ResNet-101', 33.2],
'RetinaNet500' : ['trainval35k','ResNet-101', 34.4],
'RetinaNet800' : ['trainval35k','ResNet-101-FPN', 39.1],
'M2Det512' : ['trainval35k','ResNet-101', 38.8],
'M2Det800' : ['trainval35k','VGG16',41.0],
'RefineDet320+' : ['trainval35k','ResNet-101', 38.6],
'RefineDet512+' : ['trainval35k','ResNet-101', 41.8],
'FPN' : ['trainval35k', 'ResNet101',39.8],
'NAS-FPNr' : ['trainval35k','RetinaNet', 40.5],
'NAS-FPNa' : ['trainval35k','AmoebaNet', 48.0],
'Granulated CNN' : ['trainval35k','ResNet-101', 32.0],
}

for key in models:
    print(key, '&', models[key][0], '&', models[key][1], '&', models[key][2], '\\\\')