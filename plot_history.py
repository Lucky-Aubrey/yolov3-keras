# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:36:00 2022

@author: Lucky
"""
import matplotlib.pyplot as plt
import numpy as np

# Training with freezing step
# file_name = 'yolov3_sound_my_loss_pre'
# file_name = 'history/' + file_name + '.npy'
# history_pre = np.load(file_name, allow_pickle='TRUE').item()


# # noise
# file_name = 'model_scratch_original'
# file_name = 'model_scratch_noise'
file_name = 'model_darknet_original'

# VOC
# file_name = 'VOC_darknet_original'
# file_name = 'VOC_frozenDarknet_original'


file_name = 'history/' + file_name + '.npy'
history = np.load(file_name, allow_pickle='TRUE').item()

folder = 'C:/Users/Lucky/Documents/Projektarbeit/Latex-Vorlage/Latex-Vorlage/graphics/bilder/'

# for a in history:
#     history[a] = history_pre[a] + history[a]

keys = list(history.keys())

# keys1 = keys[0]
# keys2 = keys[6]
# keys = [keys1]+[keys2]

# keys1 = keys[0:6]
# keys2 = keys[6:11]
# keys = keys1 + keys




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



figure = plt.figure()
# plt.title(file_name)
for key in [keys[0],keys[4]]:
    plt.plot(np.arange(1,31),np.array(history[key]))
plt.legend(['train','val'])
plt.ylim(bottom=-5, top=100)
# plt.xlim(0,12)
plt.xlabel("Epochs")
plt.ylabel(r'Loss $\mathcal{L}$')


# figure export setup (size, resolution)
width_, height_, resolution = 8, 6, 300
figure.set_size_inches(width_*0.3937, height_*0.3937)  # this is only inches. convert cm to inch by * 0.3937007874

import re

file_name = re.search('history/(.*).npy', file_name).group(1)
# figure export as impage and png image of certain size and resolution
plt.savefig(folder + file_name + '.pdf', dpi=resolution, bbox_inches="tight")  # bbox_inches takes care of keeping everything inside the frame that is being exported
# plt.savefig("sample.pdf", dpi=resolution, bbox_inches="tight")

plt.show()
