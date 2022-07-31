import glob

file_list = glob.glob("data/sound_evaluation/images/*/*.png")

# Load labels
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
        npy_list.append(np.array([[-1.,0.,0.,0.,0.]]))
        
npy_list = np.array(npy_list, dtype=object)

# Images
import tensorflow as tf
import matplotlib.pyplot as plt

images = tf.data.Dataset.from_tensor_slices(
    [plt.imread(a)[...,:3] for a in file_list]
    )

def resize_img(image):
    return tf.image.resize(image, [416,416])

images = images.map(resize_img)

#%% Test npy


#%%

import config

file_name = 'yolov3_2'
img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.EPOCHS
batch_size=config.BATCH_SIZE
learning_rate = config.LEARNING_RATE # TODO

#%% Model
from model import build_model
# from weight_converter_model import build_model

# # VOC2012
# num_classes = 20
# Sound
num_classes = 4
#
yolo = build_model(num_classes)

# yolo.load_weights('weights/yolov3_transferlearning_1.h5')
# yolo.load_weights('weights/yolov3_voc2012.h5')
# yolo.load_weights('checkpoints/yolov3_voc2012_github_loss5.tf')
# yolo.load_weights('checkpoints/yolov3_voc2012_github_loss2.tf')
# yolo.load_weights('checkpoints/yolov3_voc2012_my_loss_pre3.tf')
yolo.load_weights('checkpoints/yolov3_sound_my_loss4.tf')
# yolo.load_weights('checkpoints/yolov3_sound_from_scratch16.tf')

#%%

bs = 16 # Batch size: works only for bs > 1 # TODO implement for bs = 1
anchors = config.ANCHORS

evaluation_table = {0: {'TP':0, 'FP':0, 'FN':0},
                    1: {'TP':0, 'FP':0, 'FN':0},
                    2: {'TP':0, 'FP':0, 'FN':0},
                    3: {'TP':0, 'FP':0, 'FN':0}}
import numpy as np
import time
from yolo_utils import intersectionOverUnion
start = time.time()
img_i = 0
for i, img in enumerate(images.batch(bs)):
    # Prediction
    output = yolo(img)
    # True labels
    target = npy_list[i*bs:(i+1)*bs]
    
    #
    from yolo_utils import yoloBoxes, _nms
    
    # [BOXES/CONF/CLASS][BS][GRID_Y][GRID_X][ANCHORS][...]
    y1 = yoloBoxes(output[0], anchors[0], num_classes)
    y2 = yoloBoxes(output[1], anchors[1], num_classes)
    y3 = yoloBoxes(output[2], anchors[2], num_classes)
    #[GRID_SCALE][BOXES/CONF/CLASS][BS][GRID_Y][GRID_X][ANCHORS][...]
    outputs=(y1,y2,y3)
    # TODO save outputs into a textfile for looping over score_thresholds
    #[BS][BOXES/SCORES/CLASSES/VALID_DETECTIONS][...]
    result = _nms(outputs, \
         num_classes, \
         max_output_size=100, \
         iou_threshold=0.5, \
         score_threshold=0.5, \
         soft_nms_sigma=0.)
        
    for bs_i, res in enumerate(result):
        # get predictions and true values
        # [[x1,y1,x2,y2],...],scores,[class,...],....
        boxes, scores, classes, valid_detections = res
        # [class, x1, y1, x2, y2]
        labels = target[bs_i]
        # list boxes for each class separate in dictionary
        pred_boxes = {0: [],
                      1: [],
                      2: [],
                      3: []}
        true_boxes = {0: [],
                      1: [],
                      2: [],
                      3: []}
        # put boxes inside dictionaries
        for b, c in zip(boxes[0],classes[0]):
            pred_boxes[int(c)].append(b)
        # print('\nLabels:') # TODO delete
        for label in labels:
            if label[0] != -1:
                true_boxes[label[0]].append(label[1:5])
        #     print(f'class: {label[0]}')
        #     print(f'box: {label[1:5]}')
        # print('')
            
        for class_key in pred_boxes:
            j = 0
            while j < len(pred_boxes[class_key]):
                pred_box = pred_boxes[class_key][j]
                k = 0 # TODO 
                if len(true_boxes[class_key]) == 0: 
                    j += 1
                    continue
                while k < len(true_boxes[class_key]):
                    true_box = true_boxes[class_key][k]
                    if intersectionOverUnion(true_box, pred_box) > 0.5:
                        true_boxes[class_key].pop(k)
                        pred_boxes[class_key].pop(j)
                        evaluation_table[class_key]['TP'] += 1
                    else:
                        k += 1
                j += 1
            evaluation_table[class_key]['FP'] += len(pred_boxes[class_key])
            evaluation_table[class_key]['FN'] += len(true_boxes[class_key])
            
name_dict = {
    0: 'artefact',
    1: 'click',
    2: 'squeal',
    3: 'wirebrush'
    }

check0 = time.time()
print(f'Prediction of {len(images)} images using batch size of {bs}: {check0-start}s')
for e in evaluation_table:
    print(name_dict[e], evaluation_table[e])
#%% Evaluate Table

for key in evaluation_table:
    TP = evaluation_table[key]['TP']
    FP = evaluation_table[key]['FP']
    FN = evaluation_table[key]['FN']
    recall=  TP / (TP+FN) if TP+FN > 0 else None
    precision = TP / (TP+FP) if TP+FP > 0 else None
    f1 = 2 * recall * precision / (recall + precision) \
        if recall != None or precision != None else None
    evaluation_table[key].update({'recall': round(recall,2)})
    evaluation_table[key].update({'precision': round(precision,2)})
    evaluation_table[key].update({'f1': round(f1,2)})
    
for key in evaluation_table:
    print(name_dict[key],':', evaluation_table[key])

#%%

import json
with open('evaluation_detection/yolov3_sound_my_loss4_iou0_5_score0_5.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_table, f, ensure_ascii=False, indent=4)
    

with open('evaluation_detection/yolov3_sound_my_loss4_iou0_5_score0_5.json')\
    as f:
        table = json.load(f)
        
import pandas as pd
table = pd.DataFrame(table)
table = table.transpose()
print(table.to_latex(index=False))