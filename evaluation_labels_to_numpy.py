'''
This script transforms the labels of the evaluation set, which
are in .xml format, to numpy arrays and saves them to .npy files.
'''
import xmltodict
import numpy as np

folder = 'data/sound_evaluation/labels/'
name_dict = {
    'dyno': 0,
    'klacken': 1,
    'squeal': 2,
    'wirebrush': 3
    }

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
#%%

for file in onlyfiles:
    fileptr = open(f'data/sound_evaluation/labels/{file}',"r")
     
    #read xml content from the file
    xml_content= fileptr.read()

    #change xml format to ordered dict
    annotations=xmltodict.parse(xml_content,attr_prefix='',dict_constructor=dict)


    size = annotations['annotation']['size']
    size = [int(size['width']), int(size['height'])]
    boxes = annotations['annotation']['object']
    
    if type(boxes) != list:
        boxes = [boxes]
        
    label_list = []
    for box in boxes:
        name = box['name']
        bndbox = box['bndbox']
        box_x = [int(bndbox['xmin']),int(bndbox['xmax'])]
        box_x = np.array(box_x)/size[0]
        box_y = [int(bndbox['ymin']),int(bndbox['ymax'])]
        box_y = np.array([int(bndbox['ymin']),int(bndbox['ymax'])])/size[1]
        bndbox = np.array([box_x[0],box_y[0],box_x[1],box_y[1]], dtype=np.float32)
        label = np.insert(bndbox,0,name_dict[name])
        label_list.append(label)

    label_list = np.array(label_list)
    numpy_file = file.removesuffix('.xml')
    np.save(f'data/sound_evaluation/labels_numpy/{numpy_file}', label_list)
