# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:52:38 2022

@author: Lucky
"""

# =============================================================================
# Config-file
# =============================================================================
import numpy as np
IMG_SIZE = 416 # Should be multiple of 32
GRID_LIST = [IMG_SIZE // 32, IMG_SIZE // 16, IMG_SIZE // 8]
NUM_CLASSES = 4 # Animals
# NUM_CLASSES = 20 # voc2012
WEIGHTS = [1,1,5,1,1] # xy, wh, obj, noobj, class
LEARNING_RATE = 0.001 
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # 13
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # 26
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # 52
]

# ANCHORS = np.array([
#     [(116,90), (156,198), (373,326)],  # 13
#     [(30,61), (62,45), (59,119)], # 26
#     [(10,13), (16,30), (33,23)], # 52
# ])/416

# TODO multiple trainings
# # train.py: hyperparameters are stored as list to enable multiple training
# EPOCHS=[200,400]
# BATCH_SIZE=[16]

EPOCHS=50
BATCH_SIZE=16

# test.py
TEST_EPOCHS = 1
TEST_BATCH_SIZE = 1