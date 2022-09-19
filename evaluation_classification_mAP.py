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
import tensorflow as tf

# # VOC2012
# num_classes = 20
# Sound
num_classes = 4
#
yolo = build_model(num_classes)


# Pre-loaded, COCO anchors
# file = 'checkpoints/yolov3_sound_my_loss4.tf'
# Scratch, COCO anchors
# file = 'checkpoints/yolov3_sound_from_scratch16.tf'
# Scratch. hand-picked anchors
file = 'checkpoints/yolov3_scratch_noiseAnchors_.tf'

yolo.load_weights(file)

#%% load data

# Load voc2012 dataset
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
import numpy as np

tfrecord = './data/sound_val.tfrecord'
classes = './data/sound.names'
size = 416
anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])



# dataset = ds.load_tfrecord_dataset(tfrecord, classes, size)

dataset = tf.data.Dataset.list_files('data/sound_evaluation/images/*/*.png',
                                     shuffle = False)
from yolo_utils import process_image_for_classification, scale

val_dataset = dataset.map(process_image_for_classification).map(scale)

#%%

bs = 16 # Batch size: works only for bs > 1 # TODO implement for bs = 1
anchors = config.ANCHORS

name_dict = {
    0: 'artefact',
    1: 'click',
    2: 'squeal',
    3: 'wirebrush'
    }

table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(['artefact', 'click', 'squeal', 'wirebrush']),
        values=tf.constant([0, 1, 2, 3]),
    ),
    default_value=tf.constant(-1),
    name="class_weight"
)


import numpy as np
import time
start = time.time()

output_list = []
target_list = []
i = 0
for x, y in val_dataset.batch(bs):
    i+=1
    print(i,len(val_dataset.batch(bs)))
    output = yolo(x)
    #
    from yolo_utils import yoloBoxes, _nms
    y1 = yoloBoxes(output[0], anchors[0], num_classes)
    y2 = yoloBoxes(output[1], anchors[1], num_classes)
    y3 = yoloBoxes(output[2], anchors[2], num_classes)
    #[GRID_SCALE][BOXES/CONF/CLASS][BS][GRID_Y][GRID_X][ANCHORS][...]
    outputs=(y1,y2,y3)
    output_list.append(outputs)
    target_list.append(y)

#%%

iou_threshold_list = [0.5]
score_threshold_list = [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# score_threshold_list = [0.5]   
ap_results = {}
img_i = 0
for iou_threshold in iou_threshold_list:
    ap_results[iou_threshold] = {}
                                  
    ap_dict = {
        'squeal': {'recall': [], 'precision': []},
        'wirebrush': {'recall': [], 'precision': []},
        'click':  {'recall': [], 'precision': []},
        'artefact': {'recall': [], 'precision': []}
        }
    for score_threshold in score_threshold_list:
        prediction_table = []
        for o,y in zip(output_list,target_list):
            # TODO save outputs into a textfile for looping over score_thresholds
            #[BS][BOXES/SCORES/CLASSES/VALID_DETECTIONS][...]
            result = _nms(o, \
                 num_classes, \
                 max_output_size=100, \
                 iou_threshold=iou_threshold, \
                 score_threshold=score_threshold, \
                 soft_nms_sigma=0.)
                
            for bs_i, res in enumerate(result):
                boxes, scores, classes, valid_detections = res
                vd = valid_detections[0]
                if vd == 0:
                    unique_classes = []
                else:
                    unique_classes, _ = tf.unique(
                        tf.squeeze(
                            tf.stack(classes[:vd],0),0))
            
                compare_table = {
                    0: ['F','F'],
                    1: ['F','F'],
                    2: ['F','F'],
                    3: ['F','F'],
                    }
                true_class = int(table.lookup(y[bs_i]))
                if true_class != -1:
                    compare_table[true_class][1] = 'T'
                
                for c in unique_classes:
                    pred_class = int(c)
                    compare_table[pred_class][0] = 'T'
                    
                prediction_table.append(compare_table)
                img_i+=1
        
            
        # Evaluate Table
        print(len(prediction_table))
        res_table = {
            'squeal': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
            'wirebrush': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
            'click': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
            'artefact': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
            }
        
        tpi = 0
        for pred in prediction_table:
            for key in pred:
                comparison = pred[key][0]+pred[key][1]
                table_key = name_dict[key]
                if comparison == 'TT':
                    res_table[table_key]['TP'] +=1
                elif comparison == 'TF':
                    res_table[table_key]['FP'] +=1
                elif comparison == 'FT':
                    res_table[table_key]['FN'] +=1
                elif comparison == 'FF':
                    res_table[table_key]['TN'] +=1
                    
        
        for key in res_table:
            TP = res_table[key]['TP']
            FP = res_table[key]['FP']
            FN = res_table[key]['FN']
            TN = res_table[key]['TN']
            recall =  TP / (TP+FN) if TP+FN > 0 else None
            precision = TP / (TP+FP) if TP+FP > 0 else None
            if recall != None and precision != None:
                ap_dict[key]['recall'].append(recall)
                ap_dict[key]['precision'].append(precision)

    
    for key in ap_dict:
        precisions = ap_dict[key]['precision']
        recalls = ap_dict[key]['recall']
        
        # sort by recall
        recalls, precisions = list(zip(*sorted(zip(recalls,precisions))))
        recalls = list(recalls)
        precisions = list(precisions)
        # add entry for (recall = 0, precision = precisions[0]
        recalls.insert(0, 0.)
        precisions.insert(0, precisions[0])
        
        ap_dict[key]['precision'] = precisions
        ap_dict[key]['recall'] = recalls
        
        ap = np.trapz(precisions, recalls)
        # ap_dict[f'{key}']['AP'] = ap
        ap_results[iou_threshold][key] = ap
        
        
        # print(f'AP for {key}: {ap}')
    
    # mAP = sum([ap_dict[f'{key}']['AP'] for key in ap_dict])/len(ap_dict)
    mAP = sum([ap_results[iou_threshold][key] for key in ap_dict])/len(ap_dict)
    ap_results[iou_threshold]['mAP'] = mAP
    # print(f'mAP: {mAP}')
    
    # print(ap_dict)

# #%% round
# for iou in ap_results:
#     for key in ap_results[iou]:
#         ap_results[iou][key] = round(ap_results[iou][key], 2)
# print(ap_results)
#%% Save as json file
import json
import re

file = re.search('checkpoints/(.*).tf', file).group(1)
file_name = file +'_mAP'
with open(f'evaluation_classification/{file_name}.json','w+') as f:
    json.dump(ap_results, f, indent=4)
    
# save prc curve
file_name = file +'_prc'
with open(f'evaluation_classification/{file_name}.json','w+') as f:
    json.dump(ap_dict, f, indent=4)