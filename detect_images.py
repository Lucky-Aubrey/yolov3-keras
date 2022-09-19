import config

img_size = config.IMG_SIZE # Should be multiple of 32
grid_list = config.GRID_LIST # Basically this is img_size / 32
num_classes = config.NUM_CLASSES
weights = config.WEIGHTS
anchors = config.ANCHORS
epochs=config.EPOCHS
batch_size=config.BATCH_SIZE
learning_rate = config.LEARNING_RATE # TODO

#%% Model
import matplotlib.pyplot as plt
from model import build_model
# from weight_converter_model import build_model

#
yolo = build_model(num_classes)

yolo.load_weights('checkpoints/yolov3_sound_my_loss4.tf')
# yolo.load_weights('checkpoints/yolov3_sound_from_scratch16.tf')
# yolo.load_weights('checkpoints/yolov3_scratch_noiseAnchors_16.tf')
# Load voc2012 dataset

import numpy as np

#%%
tfrecord = './data/sound_val.tfrecord'
classes = './data/sound.names'
size = 416
# anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
#                          (59, 119), (116, 90), (156, 198), (373, 326)],
#                         np.float32) / 416
anchors = config.ANCHORS
anchors = np.array(anchors[1]+anchors[1]+anchors[0], dtype=np.float32)
anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


import numpy as np
import matplotlib.pyplot as plt 
import cv2
from yolo_utils import yoloBoxes, _nms
from draw_functions import draw_output

im_folder = 'data/test_images/'
name_list = {
    'squeal': '(a) squeal',
    'wirebrush': '(b) wirebrush',
    'click': '(c) click',
    'artefact': '(d) artefact',
    'multiclass': '(e) multi-class',
    'no_sound': '(f) no sound',
    }

plt.style.use("default")  # dark_background

# set LaTeX font
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "font.size": 13,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}
plt.rcParams.update(tex_fonts)

for name in name_list:
    input_img = plt.imread(im_folder + name + '.png')[...,:3]
    input_img = input_img / np.max(input_img)
    img = cv2.resize(input_img, (416,416), interpolation = cv2.INTER_AREA)
    img = img
    img = np.expand_dims(img, axis = 0)
    
    output = yolo(img)
    #
    anchors = config.ANCHORS
    y1 = yoloBoxes(output[0], anchors[0], num_classes)
    y2 = yoloBoxes(output[1], anchors[1], num_classes)
    y3 = yoloBoxes(output[2], anchors[2], num_classes)
    outputs=(y1,y2,y3)
    
    boxes, scores, classes, valid_detections = _nms(outputs, \
          num_classes, \
          max_output_size=100, \
          iou_threshold=0.5, \
          score_threshold=0.5, \
          soft_nms_sigma=0.)
        
    vd = valid_detections[0]
    
    
    img = input_img.copy()
    
    figure = plt.figure()
    
    img = draw_output(img, boxes, scores, classes, valid_detections)
    
    plt.title(f'{name_list[name]}')
    plt.xlabel(r'duration $[s]$')
    plt.ylabel(r'frequency [kHz]')
    
    w, h = img.shape[1], img.shape[0]
    y_ticks = np.arange(0,h+1, h/5, dtype=np.int16)
    y_values = np.arange(1,16+1,3)
    plt.yticks(y_ticks, y_values)
    
    x_ticks = np.arange(0,w+1, w/5-4, dtype=np.int16)
    x_values = np.arange(0,11+1,10/5, dtype=np.int16)
    plt.xticks(x_ticks, x_values)
    
    # plt.xlim(left = -0.04, right=1.04)
    # plt.ylim(bottom = -0.04, top=1.04)
    
    # figure export setup (size, resolution)
    width_, height_, resolution = 8, 6, 300
    figure.set_size_inches(width_*0.3937, height_*0.3937)  # this is only inches. convert cm to inch by * 0.3937007874
    
    # figure export as impage and png image of certain size and resolution
    # plt.savefig("sample.pdf", dpi=resolution, bbox_inches="tight")
    
    # folder = r'C:\Users\Lucky\Documents\Projektarbeit\Latex-Vorlage\Latex-Vorlage\graphics\bilder'
    # plt.savefig(folder + f'\\detect_{name}.png', dpi=resolution, bbox_inches="tight")
    # plt.imsave(folder + f'\\detect_{name}1.png', img)
    
    plt.imshow(np.flip(img,0), origin='lower')

    print(classes)
    
# import cv2
# cv2.imwrite('output.png', img)

# #%%

# # Output of detect.py
# import numpy as np
# for i, x in enumerate(output):
#     np.save(f'test_outputs/train_out_{i}', np.array(output[i]))

# # voc2012 dataset
# for i, x in enumerate(output):
#     np.save(f'test_outputs/train_true_{i}', np.array(targets[i]))
