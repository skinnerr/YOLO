import pandas as pd
import os
from os import getcwd
from os.path import join

sets=[('object-detection-crowdai', 'train', 'labels_crowdai.csv')]

classes = ['Car', 'Truck', 'Pedestrian', 'Street Lights`']


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(data_source, image_file, objects_df):
    out_file = open('udacitydata/%s/labels/%s.txt' % (data_source, image_file), 'w')

    # files are all of uniform size
    w = int(1920)
    h = int(1200)
    for index, row in objects_df.iterrows():
        cls = row['Label']
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        b = (float(row['xmin']), float(row['xmax']),
             float(row['ymin']), float(row['ymax']))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

for data_source, image_set, label_file in sets:
    in_file_df = pd.read_csv(label_file)
    directory_name = 'udacitydata/%s/labels/' % data_source
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    list_file = open('%s_%s.txt' % (data_source, image_set), 'w')
    for image_file, objects_df in in_file_df.groupby('Frame'):
        list_file.write('%s/udacitydata/%s/%s\n' % (wd, data_source, image_file))
        convert_annotation(data_source, image_file, objects_df)
    list_file.close()

#os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")

