# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 14:45:03 2022

@author: Lucky
"""
import json
import matplotlib.pyplot as plt

files = []
files.append('evaluation_detection/yolov3_sound_my_loss4_mAP.json')
files.append('evaluation_detection/yolov3_sound_from_scratch16_mAP.json')
files.append('evaluation_detection/yolov3_scratch_noiseAnchors__mAP.json')


data = []
for file in files:
    with open(file, 'r') as f:
        data.append(json.load(f))
    



plt.style.use("default")  # dark_background

# set LaTeX font
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "font.size": 13,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}
plt.rcParams.update(tex_fonts)

letter = {
    'squeal': 'a',
    'wirebrush': 'b',
    'click': 'c',
    'artefact': 'd',
    }
for key in letter:
        
    figure = plt.figure()  # get figure object
    
    

    i = 0
    color = ['blue', 'red', 'black']
    for d in data:
        precision = d['0.5'][key]['precision']
        recall = d['0.5'][key]['recall']
        plt.plot(recall, precision, color = color[i])
        i+=1
    # plt.plot(x, y**2, color='red')
    # plt.plot(x, y**2, color='black')
    plt.title(f'({letter[key]}) class {key}')
    plt.legend(['model 1', 'model 2', 'model 3'])
    plt.xlabel(r'recall')
    plt.ylabel(r'precision')
    
    plt.xlim(left = -0.04, right=1.04)
    plt.ylim(bottom = -0.04, top=1.04)
    
    # figure export setup (size, resolution)
    width_, height_, resolution = 8, 6, 300
    figure.set_size_inches(width_*0.3937, height_*0.3937)  # this is only inches. convert cm to inch by * 0.3937007874
    
    # figure export as impage and png image of certain size and resolution
    # plt.savefig("sample.pdf", dpi=resolution, bbox_inches="tight")
    
    folder = r'C:\Users\Lucky\Documents\Projektarbeit\Latex-Vorlage\Latex-Vorlage\graphics\bilder'
    plt.savefig(folder + f'\\prc2_{key}.pdf', dpi=resolution, bbox_inches="tight")


# # show in console
# plt.show()