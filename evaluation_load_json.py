import json
import pandas as pd

f0 = open('evaluation/yolov3_sound_my_loss4_iou0_5_score0_5.json')
f1 = open('evaluation/yolov3_sound_from_scratch16_iou0_5_score0_5.json')

table = json.load(f0)
table = pd.DataFrame(table).transpose()
print('Pretrained:')
print(table)
table = json.load(f1)
table = pd.DataFrame(table).transpose()
print('\nFrom Scratch:')
print(table)
