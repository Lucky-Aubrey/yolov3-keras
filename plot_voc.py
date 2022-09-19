# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:51:12 2022

@author: Lucky
"""

import yolov3_tf2.dataset as dataset
import config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def get_dataset():
    train_dataset = './data/voc2012_train.tfrecord' # TODO 
    classes = './data/voc2012.names'
    size = 416
    train_dataset = dataset.load_tfrecord_dataset(
        train_dataset, classes, size)
    return train_dataset

if __name__ == "__main__":
    
    train = get_dataset()
    
    train_boxes = []
    boxes_dict = {
        'objects': [], # objects
        'anchors': [],  # anchors
        }
    color_dict = {
        'objects': 'red', # objects
        'anchors': 'black'  # anchors
        }
    for x,y in train:
        for b in y:
            if tf.reduce_sum(b) == 0:
                break
            box = b.numpy()
            w = box[2]-box[0]
            h = box[3]-box[1]
            boxes_dict['objects'].append([w,h])
            #%%
    boxes_dict['anchors'] = [
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
        if key != 'anchors':
            plt.scatter(w,h, 10, marker='o', color=color_dict[key], alpha=0.1)
        else:
            plt.scatter(w,h, 100 ,marker='o', color=color_dict[key])
            
    plt.legend(['objects','anchors'], bbox_to_anchor = (1.0, 0.7))
    plt.xlabel(r'width $w$')
    plt.ylabel(r'height $h$')
    
    # # figure export setup (size, resolution)
    width_, height_, resolution = 8, 6, 300
    figure.set_size_inches(width_*0.3937, height_*0.3937)  # this is only inches. convert cm to inch by * 0.3937007874
    folder = r'C:\Users\Lucky\Documents\Projektarbeit\Latex-Vorlage\Latex-Vorlage\graphics\bilder'
    plt.savefig(folder + "\\voc2012.pdf", dpi=resolution, bbox_inches="tight")

            
    