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
        0: 0, # artefact
        1: 0, # click
        2: 0, # squeal
        3: 0,  # wirebrush
        4: 0,  # multi-class
        5: 0,  # silent
        }
    
    names_dict = {
        0: 'artefact',
        1: 'click',
        2: 'squeal',
        3: 'wirebrush',
        4: 'multi_class',
        5: 'silent',
        }
    
    for x,y in train:
        classes=[]
        for b in y:
            if tf.reduce_sum(b) == 0:
                break
            box = b.numpy()
            classes.append(int(box[4]))
        classes=list(set(classes))
        if len(classes) == 0:
            boxes_dict[5] +=1
        elif len(classes) == 1:
            boxes_dict[classes[0]] +=1
        else:
            boxes_dict[4] += 1
            
    sum_objects = 0
    for key in boxes_dict:
        print(names_dict[key], boxes_dict[key])
        sum_objects += boxes_dict[key]
    print('sum', sum_objects)
