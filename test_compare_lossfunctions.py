from yolov3_tf2.dataset import (transform_images, 
                                load_tfrecord_dataset, 
                                transform_targets)
import numpy as np
import tensorflow as tf
# =============================================================================
# tfrecord = './data/voc2012_val.tfrecord'
# classes = './data/voc2012.names'
# size = 416
# anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#                          (59, 119), (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 416
# anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
# 
# dataset = load_tfrecord_dataset(
#             tfrecord, classes, size)
# dataset = dataset.shuffle(512)
# img_raw, _label = next(iter(dataset.take(1)))
# 
# img = tf.expand_dims(img_raw, 0)
# img = transform_images(img, size)
# 
# _label = tf.expand_dims(_label, 0)
# targets = transform_targets(_label, anchors, anchor_masks, size)
# =============================================================================

# =============================================================================
# # Animal dataset
# np.save('test_outputs/train_true13_0',
#         np.array(tf.expand_dims(y_train[i][0],0)))
# np.save('test_outputs/train_true13_1',
#         np.array(tf.expand_dims(y_train[i][1],0)))
# np.save('test_outputs/train_true13_2',
#         np.array(tf.expand_dims(y_train[i][2],0)))
# =============================================================================


# =============================================================================
# # Load predictions and targets after execution of detect.py
# y_true0 = np.load('test_outputs/train_true_0.npy')
# y_pred0 = np.load('test_outputs/train_out_0.npy')
# y_true1 = np.load('test_outputs/train_true_1.npy')
# y_pred1 = np.load('test_outputs/train_out_1.npy')
# y_true2 = np.load('test_outputs/train_true_2.npy')
# y_pred2 = np.load('test_outputs/train_out_2.npy')
# =============================================================================


    
# =============================================================================
# # Datatype
# y_true0 = tf.cast(y_true0, tf.float32)
# y_pred0 = tf.cast(y_pred0, tf.float32)
# y_true1 = tf.cast(y_true1, tf.float32)
# y_pred1 = tf.cast(y_pred1, tf.float32)
# y_true2 = tf.cast(y_true2, tf.float32)
# y_pred2 = tf.cast(y_pred2, tf.float32)
# 
# =============================================================================



# My Loss function
from model import yoloLoss

anchors = np.array([
    [(116,90), (156,198), (373,326)],  # 13
    [(30,61), (62,45), (59,119)], # 26
    [(10,13), (16,30), (33,23)], # 52
])/416
weights = [1,1,1,1,1]
num_classes = 20 # voc2012

grids = (13,26,52)

# =============================================================================
# y_true_one_hot = []
# y_pred_one_hot = []
# y_true_github = []
# y_pred_github = []
# for grid in grids:
#     # -------MyPart--------
#     # outputs and targets with one hot label for my loss function
#     l_one_hot = np.zeros((1,grid,grid,3,5+num_classes),dtype= np.float32)
#     # Class 1: -> '2nd' one
#     l_one_hot[0,3,3,0] = np.array(
#         [1.,0.5,0.5,0.4,0.4] + [0] + [1] + [0] * (num_classes-2), dtype=np.float32)
#     
#     y_true_one_hot.append(tf.constant(l_one_hot,dtype=tf.float32))
# 
#     l_one_hot = np.zeros((1,grid,grid,3,5+num_classes),dtype= np.float32)
#     # l_one_hot = np.ones((1,grid,grid,3,5+num_classes),dtype= np.float32)* (-5)
#     # l_one_hot = np.ones((1,grid,grid,3,5+num_classes),dtype= np.float32)* (-10)
#     
#     y_pred_one_hot.append(tf.constant(l_one_hot,dtype=tf.float32))
#     
#     # ------Github--------
#     # outputs and targets with label encoding for github loss function
#     l_label = np.zeros((1,grid,grid,3,6),dtype= np.float32)
#     # Class 1: -> 1.
#     l_label[0,3,3,0] = np.array(
#         [0.3,0.3,0.7,0.7,1.,1.], dtype=np.float32)
#     
#     y_true_github.append(tf.constant(l_label,dtype=tf.float32))
#     
#     l_label = np.zeros((1,grid,grid,3,5+num_classes),dtype= np.float32)
#     
#     y_pred_github.append(tf.constant(l_label,dtype=tf.float32))
# =============================================================================

y_true_one_hot = []
y_pred_one_hot = []
for i in list(range(3)):
    y_true_one_hot.append(
        tf.cast(np.load(f'test_outputs/train_true_{i}.npy'), tf.float32))
    y_pred_one_hot.append(
        tf.cast(np.load(f'test_outputs/train_out_{i}.npy'), tf.float32))
   

y_true_github = y_true_one_hot
y_pred_github = y_pred_one_hot

loss0 = yoloLoss(anchors[0], weights, num_classes)
loss1 = yoloLoss(anchors[1], weights, num_classes)
loss2 = yoloLoss(anchors[2], weights, num_classes)

l0 = loss0(y_true_one_hot[0],y_pred_one_hot[0])
l1 = loss1(y_true_one_hot[1],y_pred_one_hot[1])
l2 = loss2(y_true_one_hot[2],y_pred_one_hot[2])

print('\n My loss function:')
print('\n Individiual losses: out_0, out_1, out_2')
print(np.array((l0, l1, l2)))
print('\n All losses: ')
print(np.array(l0+l1+l2))
print('')

#%%
# Github loss function
from github_model import YoloLoss

anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                          (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
loss_function = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]

loss0 = loss_function[0]
loss1 = loss_function[1]
loss2 = loss_function[2]
l0 = loss0(y_true_github[0],y_pred_github[0])
l1 = loss1(y_true_github[1],y_pred_github[1])
l2 = loss2(y_true_github[2],y_pred_github[2])

print('\n Github Loss function:')
print('\n Individiual losses: out_0, out_1, out_2')
print(np.array((l0, l1, l2)))
print('\n All losses: ')
print(np.array(l0+l1+l2))

