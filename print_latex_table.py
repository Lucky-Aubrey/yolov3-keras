import json

# file = 'evaluation_classification/yolov3_sound_my_loss4_iou0_5_score0_5.json'
# file = 'evaluation_classification/yolov3_sound_from_scratch16_iou0_5_score0_5.json'
file = 'evaluation_classification/yolov3_scratch_noiseAnchors_.json'
with open(file, 'r') as f:
    data = json.load(f)
# file2 = 'evaluation_classification/yolov3_sound_my_loss4_mAP.json'
# file2 = 'evaluation_classification/yolov3_sound_from_scratch16_mAP.json'
file2 = 'evaluation_classification/yolov3_scratch_noiseAnchors__mAP.json'
with open(file2, 'r') as f:
    data2 = json.load(f)

print('\\hline')
head = 'category'
for key in data['squeal']:
    head = head + ' & ' + key
    
head = head + ' & ' + 'AP' + ' & ' + 'mAP'
head = head + ' \\\\'

print(head)
print('\\hline')

for key in data:
    row = ' & ' + key
    if key == 'squeal':
        row = '1' + row
    for val in data[key]:
        row = row + ' & ' + str(data[key][val])
    row = row + ' & ' + str(round(data2['0.5'][key],2))
    if key == 'squeal':
        row = row + ' & ' + str(round(data2['0.5']['mAP'],2))
    else:
        row = row + ' & '
        
    row = row + ' \\\\'
    print(row)
print('\\hline')