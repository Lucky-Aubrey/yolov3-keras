

#%% load data
import tensorflow as tf

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

# dataset = ds.load_tfrecord_dataset(tfrecord, classes, size)

dataset = tf.data.Dataset.list_files('data/sound_evaluation/images/*/*.png',
                                     shuffle = False)
from yolo_utils import process_image_for_detection, scale

# Loading dataset

val_dataset = dataset.map(process_image_for_detection).map(scale)

#%%

bs = 2 # Batch size: works only for bs > 1 # TODO implement for bs = 1
ns = 32 # Number of images
anchors = config.ANCHORS

prediction_table = []
import numpy as np
import time
start = time.time()
img_i = 0
for x, y in val_dataset.batch(bs).take(10):
    output = yolo(x)
    #
    from yolo_utils import yoloBoxes, _nms
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
    

check0 = time.time()
print(f'Prediction of {ns} images using batch size of {bs}: {check0-start}s')
# print(f'\nAmount images: {len(prediction_table)}')
# for i, a in enumerate(prediction_table):
#     print(i+1, a)
    
#%% Evaluate Table

res_table = {
    'squeal': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
    'wirebrush': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
    'click': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
    'artefact': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    }

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
    recall =  TP / (TP+FN)
    precision = TP / (TP+FP)
    f1 = 2 * recall * precision / (recall + precision)
    res_table[key]['recall'] = round(recall,2)
    res_table[key]['precision'] = round(precision,2)
    res_table[key]['f1'] =  round(f1,2)
    
for key in res_table:
    print(key,':', res_table[key])

# #%% Save as json file
# import json
# file_name = 'yolov3_sound_from_scratch16_iou0_5_score0_5'
# with open(f'evaluation/{file_name}.json','w+') as f:
#     json.dump(res_table, f, indent=4)