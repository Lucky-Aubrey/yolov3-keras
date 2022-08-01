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

# yolo.load_weights('weights/yolov3_transferlearning_1.h5')
# yolo.load_weights('weights/yolov3_voc2012.h5')
# yolo.load_weights('checkpoints/yolov3_voc2012_github_loss5.tf')
# yolo.load_weights('checkpoints/yolov3_voc2012_github_loss2.tf')
# yolo.load_weights('checkpoints/yolov3_voc2012_my_loss_pre3.tf')

# yolo.load_weights('checkpoints/yolov3_sound_my_loss4.tf')
yolo.load_weights('checkpoints/yolov3_sound_from_scratch16.tf')

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

bs = 2 # Batch size: works only for bs > 1 # TODO implement for bs = 1
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

prediction_table = []
import numpy as np
import time
start = time.time()
img_i = 0
for x, y in val_dataset.batch(bs):
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
print(f'Prediction of images using batch size of {bs}: {check0-start}s')
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