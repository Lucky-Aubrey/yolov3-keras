# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:07:31 2022

@author: Lucky
"""

import cv2
import numpy as np

# =============================================================================
# TODO create a draw function which draws the center point
# TODO clean up
# TODO documentation
# =============================================================================

def draw_box(image, box, color=(1,0,0), xywh=False):
    img = image
    thick=4
    imgHeight, imgWidth, _ = img.shape
    
    if xywh is True:
        left = int((box[0]-box[2]/2)*imgWidth)
        bottom = int((box[1]-box[3]/2)*imgHeight)
        right = int((box[0]+box[2]/2)*imgWidth)
        top = int((box[1]+box[3]/2)*imgHeight)
    else:
        left = int(box[0]*imgWidth)
        bottom = int(box[1]*imgHeight)
        right= int(box[2]*imgWidth)
        top = int(box[3]*imgHeight)
        
    cv2.rectangle(img,(left, bottom), (right, top), color=color, thickness=thick)
    return img

color_dict = {
    0: (0,255,0),
    1: (0,255,255),
    2: (255,128,0),
    3: (255,0,255)
    }
name_dict = {
    0: 'Artefact',
    1: 'Click',
    2: 'Squeal',
    3: 'Wirebrush'
    }

def draw_output(img, boxes, scores, classes, valid_detections):
    vd = valid_detections[0]
    shift_click = np.arange(0,vd*20,20)
    click_i = 0
    for box, score, p_class in zip(boxes[0,:vd], scores[0,:vd], classes[0,:vd]):
        imgHeight, imgWidth, _ = img.shape
        x1y1 = (int(box[0]*imgWidth), int(box[1]*imgHeight))
        x2y2 = (int(box[2]*imgWidth), int(box[3]*imgHeight))
        img = cv2.rectangle(img,
                      x1y1,
                      x2y2,
                      color=color_dict[int(p_class)],
                      thickness=3)
        text_coord = (x1y1[0]-2,x1y1[1]-10)
        if text_coord[1] < 10:
            text_coord = (x1y1[0]-2,x2y2[1]+15)
        if int(p_class) == 2:
            x = max(0,x1y1[0]-100)
            y = x2y2[1]-20
            text_coord = (x,y)
        if int(p_class) == 0:
            x = max(0,x1y1[0]-140)
            y = int((x1y1[1]+x2y2[1])/2)
            text_coord = (x,y)
        if int(p_class) == 1:
            x = max(0,x1y1[0]-80)
            y = int((x1y1[1]+x2y2[1])/2) + shift_click[click_i]
            text_coord = (x,y)
            click_i+=1
            
        img = cv2.putText(img, 
                          str(np.round(score,2))+' '+name_dict[int(p_class)], 
                          text_coord, 
                          cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          0.8,
                          color_dict[int(p_class)],
                          2)
    return img

def draw_cells(image, mask, color=(0,1,0)):
    grid_x = np.arange(mask.shape[0])
    grid_y = np.arange(mask.shape[1])
    
    for x in grid_x:
        for y in grid_y:
            if mask[x,y]==True:
                box = np.array([x+0.5, y+0.5, 1, 1])/13
                draw_box(image, box, color)
    return image
        
# TODO documentation
import matplotlib.pyplot as plt
def draw_check(image, y_targets, instance):
    i = instance
    img = image[i]
    t_all = y_targets
    a_len = 9
    for g_id,grid in enumerate(t_all):
        t_grid = grid[i]
        for gx,col in enumerate(t_grid):
            for gy,cell in enumerate(col):
                for a in list(range(3)):
                    if cell[a*a_len] != 0:
                        #head
                        head = cell[a*a_len:a*a_len+a_len]
                        x = (gx+head[1])/13
                        y = (gy+head[2])/13
                        w = head[3]/13
                        h = head[4]/13
                        box = [x,y,w,h]
                        img = draw_box(img, box)
    plt.imshow(img)
                        
                        
        
    