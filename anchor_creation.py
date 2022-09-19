# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 00:08:26 2022

@author: Lucky
"""
import yolov3_tf2.dataset as dataset
import config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_dataset():
    train_dataset = './data/sound_train.tfrecord' # TODO 
    classes = './data/sound.names'
    size = 416
    train_dataset = dataset.load_tfrecord_dataset(
        train_dataset, classes, size)
    return train_dataset

if __name__ == "__main__":
    
    train = get_dataset()
    
    train_boxes = []
    boxes_dict = {
        1: [], # click
        2: [], # squeal
        3: [],  # wirebrush
        0: [], # artefact
        4: [],  # COCO anchors
        5: [],  # handpickedanchors
        }
    color_dict = {
        0: 'blue', # artefact
        1: 'green', # click
        2: 'red', # squeal
        3: 'orange',  # wirebrush
        4: 'purple',  # anchors
        5: 'black'  # anchors
        }
    for x,y in train:
        for b in y:
            if tf.reduce_sum(b) == 0:
                break
            box = b.numpy()
            w = box[2]-box[0]
            h = box[3]-box[1]
            b_class = int(box[4])
            boxes_dict[b_class].append([w,h])
#%%
    # COCO
    boxes_dict[4] = [
         (0.28, 0.22),
         (0.38, 0.48),
         (0.9, 0.78),
         (0.07, 0.15),
         (0.15, 0.11),
         (0.14, 0.29),
         (0.02, 0.03),
         (0.04, 0.07),
         (0.08, 0.06),
        ]
    # Handpicked
    boxes_dict[5] = [
        [0.20, 0.08],
        [0.50, 0.12],
        [0.75, 0.08],
        [0.08, 0.60],
        [0.08, 0.80],
        [0.50, 0.40],
        [0.90, 0.45],
        [0.40, 0.93],
        [0.70, 0.93]
        ]        
    
    
    plt.style.use("default")  # dark_background
    
    
    # set LaTeX font
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    plt.rcParams.update(tex_fonts)
    
    figure = plt.figure()  # get figure object
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    for key in boxes_dict:
        w,h = np.array(boxes_dict[key]).transpose()
        if key != 4 and key!= 5:
            plt.scatter(w,h, 10, marker='x', color=color_dict[key])
        else:
            plt.scatter(w,h, 50 ,marker='o', color=color_dict[key])
            
    plt.legend(['artefact','click','squeal','wirebrush','COCO anchors','handpicked'], bbox_to_anchor = (1.0, 0.7))
    plt.xlabel(r'width $w$')
    plt.ylabel(r'height $h$')
    
    # # figure export setup (size, resolution)
    width_, height_, resolution = 8, 6, 300
    figure.set_size_inches(width_*0.3937, height_*0.3937)  # this is only inches. convert cm to inch by * 0.3937007874
    folder = r'C:\Users\Lucky\Documents\Projektarbeit\Latex-Vorlage\Latex-Vorlage\graphics\bilder'
    plt.savefig(folder + "\\noise_anchors.pdf", dpi=resolution, bbox_inches="tight")

            
    