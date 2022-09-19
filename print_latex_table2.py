import json


# file = 'evaluation_detection/yolov3_sound_my_loss4_mAP.json'
# file = 'evaluation_detection/yolov3_sound_from_scratch16_mAP.json'
file = 'evaluation_detection/yolov3_scratch_noiseAnchors__mAP.json'
with open(file, 'r') as f:
    data = json.load(f)

head = 'model & category & $\\mathrm{AP}_{0.50} & \\mathrm{AP}_{0.75}' + \
    ' & \\mathrm{AP}_{0.90} & \\mathrm{mAP}_{0.50} & \\mathrm{mAP}_{0.75}' + \
        ' & \\mathrm{mAP}_{0.90} \\\\'
print(head)
for cat in data['0.5']:
    if cat == 'mAP':
        continue
    if cat  == 'squeal':
        line = '1 & ' + cat
    else:
        line = ' & ' +cat
    tail = ''
    for iou in data:
        ap = round(data[iou][cat]['AP'],2)
        line += ' & ' + str(ap)

        if cat == 'squeal':
            mAP = round(data[iou]['mAP'],2)
            tail += ' & ' + str(mAP)
        else:
            tail += ' & '
    line += tail + '\\\\'

    print(line)
print('\\hline')
    
# head = head + ' & ' + 'AP' + ' & ' + 'mAP'
# head = head + ' \\\\'

# print(head)
# print('\\midrule')

# for key in data:
#     row = key
#     for val in data[key]:
#         row = row + ' & ' + str(data[key][val])
#     row = row + ' & ' + str(data2['0.5'][key])
#     if key == 'squeal':
#         row = row + ' & ' + str(data2['0.5']['mAP'])
#     else:
#         row = row + ' & '
        
#     row = row + ' \\\\'
#     print(row)
# print('\\bottomrule')