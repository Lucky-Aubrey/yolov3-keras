from model import yoloLoss
from config import ANCHORS, WEIGHTS, NUM_CLASSES
import tensorflow as tf

anchors = ANCHORS
weights = WEIGHTS
num_classes = NUM_CLASSES

grid = 13

loss = yoloLoss(anchors[0], weights, num_classes)

y_pred = tf.random.uniform((1,grid,grid,3,5+num_classes))
y_true = tf.zeros_like(y_pred, tf.float32)
print(loss(y_true,y_pred))
y_true = tf.ones_like(y_pred, tf.float32)
print(loss(y_true,y_pred))


#%% My loss function
num_classes = 20

np.save('test_outputs/train_out13_0', np.array(output[0]))
np.save('test_outputs/train_out13_1', np.array(output[1]))
np.save('test_outputs/train_out13_2', np.array(output[2]))
np.save('test_outputs/train_true13_0',
        np.array(targets[0]))
np.save('test_outputs/train_true13_1',
        np.array(targets[1]))
np.save('test_outputs/train_true13_2',
        np.array(targets[2]))
# np.save('test_outputs/train_true13_0',
#         np.array(tf.expand_dims(y_train[i][0],0)))
# np.save('test_outputs/train_true13_1',
#         np.array(tf.expand_dims(y_train[i][1],0)))
# np.save('test_outputs/train_true13_2',
#         np.array(tf.expand_dims(y_train[i][2],0)))

#
import numpy as np
from model import yoloLoss
from yolo_utils import yoloBoxes

loss0 = yoloLoss(anchors[0], weights, num_classes)
y_true0 = np.load('test_outputs/train_true13_0.npy')
y_pred0 = np.load('test_outputs/train_out13_0.npy')
y_true0 = tf.cast(y_true0, tf.float32)
y_pred0 = tf.cast(y_pred0, tf.float32)
l0 = loss0(y_true0,y_pred0)
#
#
loss1 = yoloLoss(anchors[1], weights, num_classes)
y_true1 = np.load('test_outputs/train_true13_1.npy')
y_pred1 = np.load('test_outputs/train_out13_1.npy')
y_true1 = tf.cast(y_true1, tf.float32)
y_pred1 = tf.cast(y_pred1, tf.float32)
l1 = loss1(y_true1,y_pred1)

loss2 = yoloLoss(anchors[2], weights, num_classes)
y_true2 = np.load('test_outputs/train_true13_2.npy')
y_pred2 = np.load('test_outputs/train_out13_2.npy')
y_true2 = tf.cast(y_true2, tf.float32)
y_pred2 = tf.cast(y_pred2, tf.float32)
l2 = loss2(y_true2,y_pred2)

print('\n')
print(l0, l1, l2)
print('\n All losses: ')
print(np.array(l0+l1+l2))


#%% Github loss function
from github_model import YoloLoss
anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                          (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
loss_function = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]

num_classes = 20

np.save('test_outputs/train_out13_0', np.array(output[0]))
np.save('test_outputs/train_out13_1', np.array(output[1]))
np.save('test_outputs/train_out13_2', np.array(output[2]))
np.save('test_outputs/train_true13_0',
        np.array(targets[0]))
np.save('test_outputs/train_true13_1',
        np.array(targets[1]))
np.save('test_outputs/train_true13_2',
        np.array(targets[2]))
# np.save('test_outputs/train_true13_0',
#         np.array(tf.expand_dims(y_train[i][0],0)))
# np.save('test_outputs/train_true13_1',
#         np.array(tf.expand_dims(y_train[i][1],0)))
# np.save('test_outputs/train_true13_2',
#         np.array(tf.expand_dims(y_train[i][2],0)))

#
import numpy as np
from model import yoloLoss
from yolo_utils import yoloBoxes

loss0 = loss_function[0]
y_true0 = np.load('test_outputs/train_true13_0.npy')
y_pred0 = np.load('test_outputs/train_out13_0.npy')
y_true0 = tf.cast(y_true0, tf.float32)
y_pred0 = tf.cast(y_pred0, tf.float32)
l0 = loss0(y_true0,y_pred0)
#
#
loss1 = loss_function[1]
y_true1 = np.load('test_outputs/train_true13_1.npy')
y_pred1 = np.load('test_outputs/train_out13_1.npy')
y_true1 = tf.cast(y_true1, tf.float32)
y_pred1 = tf.cast(y_pred1, tf.float32)
l1 = loss1(y_true1,y_pred1)

loss2 = loss_function[2]
y_true2 = np.load('test_outputs/train_true13_2.npy')
y_pred2 = np.load('test_outputs/train_out13_2.npy')
y_true2 = tf.cast(y_true2, tf.float32)
y_pred2 = tf.cast(y_pred2, tf.float32)
l2 = loss2(y_true2,y_pred2)

print('\n')
print(l0, l1, l2)
print('\n All losses: ')
print(np.array(l0+l1+l2))
