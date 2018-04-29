#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file',
	required=True)

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--input',
	nargs='+',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-t',
    '--threshold',
    help='a threshold to detect at (will override any threshold set in the config file)')

argparser.add_argument(
    '-o',
    '--output',
    help='output directory to place detected files (defaults to the image\' current directory)')

def _main_(args):
    config_path  = args.conf
    image_paths = args.input

    if image_paths is None or len(image_paths) == 0:
        print('Must specify at least one image via -i')

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    weights_path = args.weights if args.weights is not None else config['train']['saved_weights_name'] 
    threshold = args.threshold if args.threshold is not None else config['model'].get('threshold')
    output_dir = args.output if args.output is not None else None
    
    if output_dir is not None and not os.path.exists(output_dir):
	os.makedirs(output_dir)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
		threshold	    = threshold)

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    for image_path in image_paths:
        in_dir, in_name = os.path.split(image_path) 
        out_dir = in_dir if output_dir is None else output_dir
        out_file = os.path.join(out_dir, in_name[:-4] + '_detected_threshold' + str(threshold) + in_name[-4:])
        
        if image_path[-4:] == '.mp4':
                
            video_reader = cv2.VideoCapture(image_path)
    
            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    
            video_writer = cv2.VideoWriter(out_file,
                                   cv2.VideoWriter_fourcc(*'MPEG'), 
                                   50.0, 
                                   (frame_w, frame_h))
    
            for i in tqdm(range(nb_frames)):
                _, image = video_reader.read()
                
                boxes = yolo.predict(image)
                image = draw_boxes(image, boxes, config['model']['labels'])
    
                video_writer.write(np.uint8(image))
    
            video_reader.release()
            video_writer.release()  
        else:
            image = cv2.imread(image_path)
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])
    
            print(len(boxes), 'boxes are found')
    
            cv2.imwrite(out_file, image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
