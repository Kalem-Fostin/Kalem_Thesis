#!/usr/bin/env python3
# -*- coding: utf-8 -*
# ***********************************************
# @Time  : 25/02/2021 21:48
# @File  : cfg.py
# @email : xiang.chen.17@ucl.ac.uk
# @author: Xiang Chen
# ***********************************************
BATCH_SIZE = 2
NUM_CLASSES = 3 # These classes are sky, obstacle and water
EPOCH_NUMBER = 100
TRAIN_ROOT = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/train'
TRAIN_LABEL = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/train_label'
VAL_ROOT = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/val'
VAL_LABEL = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/val_label'
TEST_ROOT = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/test'
TEST_LABEL = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/test_label'
class_dict_path = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/class_dict.csv'
SEQ_TXT = '/home/bluebouy/kalem_stuff/object_detection/datasets/IMPORTANT_FOLDER_USE_IT/frame_name.txt'
SAVE_DIR = './result_pics/'
MODEL_WEIGHTS = '/home/bluebouy/kalem_stuff/object_detection/WODIS_weights.pth'  # for inference
RESUME = True
crop_size = (384,512) # image height and width for preprocessing
#IMG_SIZE = (480, 640)
IMG_SIZE = (384, 512)  # the input image size for the training, inference
