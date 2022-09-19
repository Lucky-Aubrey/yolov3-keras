import yolov3_tf2.dataset as dataset
import config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob


def get_dataset():
    train_dataset = './data/sound_train.tfrecord' # TODO 
    # train_dataset = './data/sound_val.tfrecord' # TODO 
    classes = './data/sound.names'
    size = 416
    train_dataset = dataset.load_tfrecord_dataset(
        train_dataset, classes, size)
    return train_dataset

def get_testset():
    file_list = glob.glob("data/sound_evaluation/images/*/*.png")

    # Load labels
    file_names = []
    class_list = []
    for file in file_list:
        file_names.append( file.split('\\')[-1].split('.')[0])
        class_list.append( file.split('\\')[-2])
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
            
    return npy_list

if __name__ == "__main__":
    
    # dataset = get_dataset()
    dset = get_testset()
    
    
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
    
    # for x,y in dataset:
    for y in dset:
        classes=[]
        for b in y:
            if tf.reduce_sum(b) == 0:
                break
            # box = b.numpy()
            box = b
            # classes.append(int(box[4]))
            if int(box[0]) != -1:
                classes.append(int(box[0]))
                
        classes=classes
        for c in classes:
            boxes_dict[c] += 1
            
    sum_objects = 0
    for key in boxes_dict:
        print(names_dict[key], boxes_dict[key])
        sum_objects += boxes_dict[key]
    print('sum', sum_objects)
