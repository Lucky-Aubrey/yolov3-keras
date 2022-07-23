# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 21:36:00 2022

@author: Lucky
"""
import matplotlib.pyplot as plt
import numpy as np

file_name = 'yolov3_sound_my_loss_pre'
file_name = 'history/' + file_name + '.npy'
history_pre = np.load(file_name, allow_pickle='TRUE').item()


file_name = 'yolov3_sound_my_loss'
file_name = 'history/' + file_name + '.npy'
history = np.load(file_name, allow_pickle='TRUE').item()

for a in history:
    history[a] = history_pre[a] + history[a]

keys = list(history.keys())

# keys1 = keys[0]
# keys2 = keys[6]
# keys = [keys1]+[keys2]

# keys1 = keys[0:6]
# keys2 = keys[6:11]
# keys = keys1 + keys


plt.figure()
plt.title(file_name)
for key in [keys[0],keys[4]]:
    plt.plot(np.array(history[key]))
plt.legend([keys[0],keys[4]])
plt.ylim(0,100)
plt.xlim(0,12)
plt.xlabel("Epochs")
plt.show()
