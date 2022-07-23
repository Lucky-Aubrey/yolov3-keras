'''
Labels for 44 squeal images are missing.
16 labels for squeal are unnecessarily in the test set.
This script provides help in finding them and fixing the data set.
Use makesense.ai to label for the 44 squeal images and convert to .npy
'''
import glob

file_list = glob.glob("data/sound_evaluation/images/*/*.png")

file_names = []
class_list = []
for file in file_list:
    file_names.append( file.split('\\')[-1].split('.')[0])
    class_list.append( file.split('\\')[-2])

import numpy as np
folder = 'data/sound_evaluation/corrected_labels/'
npy_list = []
except_list = []
for i, (file, true_class) in enumerate(zip(file_names, class_list)):
    if true_class != 'no_sound':
        try:
            npy_list.append(np.load(folder + file + '.npy'))
        except:
            except_list.append(file_list[i])
                
    else:
        npy_list.append(np.array([]))
    
# print(npy_list)
# print(class_list)
# for a in npy_list:
#     print(a)
for e in except_list:
    print(e)
print(f'except_list: {len(except_list)} missing data')
    
#%%
'''
Check which images are missing for the labels
'''
# Labels
file_list = glob.glob("data/sound_evaluation/corrected_labels/*.npy")

# Get label file names
file_names = []
for file in file_list:
    file_names.append( file.split('\\')[-1].split('.')[0])

# 
import numpy as np
folder = 'data/sound_evaluation/all_images/'
npy_list2 = []
except_list2 = []
from os import path
for i, file in enumerate(file_names):
    if not path.exists(folder + file + '.png'):
        except_list2.append(file_list[i])
        # print(file + '.png')

for e in except_list2:
    print(e)
print(len(except_list2))       

# Save unnecessary labels in .txt-file
# with open(r'data/sound_evaluation/unnecessary_labels.txt', 'w') as fp:
#     fp.write('\n'.join(except_list2))

#%% Use makesense.ai to generate labels for the 44 squeal images
import cv2
import matplotlib.pyplot as plt
import pandas as pd

yolo_format_labels = 'data/sound_evaluation/extra_labels/yolo_format/'
boxes = []
for e in except_list:
    txt_path = yolo_format_labels + e.split('\\')[-1]
    file_label = txt_path.removesuffix('.png')+'.txt'
    df = pd.read_csv(file_label, delimiter=' ', header=None)
    lines = df.to_numpy()
    
    corrected = []
    for l in lines:
        c = l
        c[0] = 2.
        x1 = c[1] - c[3]/2
        y1 = c[2] - c[4]/2
        x2 = c[1] + c[3]/2
        y2 = c[2] + c[4]/2
        c[1] = x1
        c[2] = y1
        c[3] = x2
        c[4] = y2
        
        corrected.append(c)
    corrected = np.array(corrected)
    boxes.append(corrected[...,1:])
    
    
    # Save Files in npy format
    # save_folder = 'data/sound_evaluation/extra_labels/numpy_format/' 
    # numpy_path = save_folder + e.split('\\')[-1].removesuffix('.png')
    # np.save(numpy_path, corrected)
    
#%%
index = 7

img = plt.imread(except_list[index])
boxes = np.array(boxes[index]) * np.array([img.shape[1],img.shape[0],
                                                img.shape[1],img.shape[0]])
boxes = np.array(boxes, dtype=int)
for box in boxes:
    x1y1 = box[:2]
    x2y2 = box[2:4]
    img = cv2.rectangle(img, x1y1, x2y2, (1.,0.,0.),2)
        
plt.imshow(img)
