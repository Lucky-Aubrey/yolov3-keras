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
# NUM_CLASSES = 4 # Noise
NUM_CLASSES = 4 # Animals
# NUM_CLASSES = 20 # voc2012
WEIGHTS = [1,1,5,1,1] # xy, wh, obj, noobj, class
LEARNING_RATE = 0.001 
EPOCHS=30
BATCH_SIZE=16
anchor_id = 'original'
mode = 'darknet' # 'scratch', 'darknet', 'frozenDarknet'

file_name = f'model_{mode}_{anchor_id}'


if anchor_id == 'original':
# Original Anchors
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],  # 13
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], # 26
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)], # 52
    ]
elif anchor_id == 'noise':

    # Hand-picked
    ANCHORS = [
        [(0.40, 0.93), (0.90, 0.45), (0.70, 0.93)], # 13x13 grid
        [(0.08, 0.60), (0.08, 0.80), (0.50, 0.40)], # 26x26 grid
        [(0.20, 0.08), (0.50, 0.12), (0.75, 0.08)], # 52x52 grid
    ]



# test.py
TEST_EPOCHS = 1
TEST_BATCH_SIZE = 1