import re
import os
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime
from functools import partial
try:
    from tf_code.auoaugment import distort_image
except Exception as e:
    print(e)

AUTO = tf.data.experimental.AUTOTUNE


def arcface_format(posting_id, image, label_group, matches):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group, matches

def arcface_inference_format(posting_id, image, label_group, matches):
    return image,posting_id

def arcface_eval_format(posting_id, image, label_group, matches):
    return image,label_group

def random_rot_shear(img, rot_limit=10, shear_limit=10):
    if rot_limit and tf.random.uniform([]) <= 0.3:
        rot_d = tf.random.uniform([], -rot_limit, rot_limit)
        img = tfa.image.rotate(img, rot_d * np.pi/180, fill_value=1.0)
    if shear_limit and tf.random.uniform([]) <= 0.2:
        shear_d = tf.random.uniform([], -shear_limit, shear_limit)
        img = tfa.image.shear_x(img, shear_d * np.pi/180, 0.0)
    return img

def gaussain_noise(image, p=0.1):
    if tf.random.uniform([]) <= p:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.1, dtype=tf.float32)
        return image + noise
    return image

# Data augmentation function
def data_augment(config, posting_id, image, label_group, matches):
    
    if config.random_crop:
        image = tf.image.random_crop(image, size=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

    if config.augname == 'normal':
        if config.CUTOUT:
            N_CUTOUT = 1
            for cutouts in range(N_CUTOUT):
                if tf.random.uniform([]) <= 0.5:
                    DIM = config.IMAGE_SIZE
                    CUTOUT_LENGTH = DIM//6
                    x1 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
                    x2 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
                    filter_ = tf.concat([tf.zeros((x1,CUTOUT_LENGTH)),tf.ones((CUTOUT_LENGTH,CUTOUT_LENGTH)),tf.zeros((DIM-x1-CUTOUT_LENGTH,CUTOUT_LENGTH))],axis=0)
                    filter_ = tf.concat([tf.zeros((DIM,x2)),filter_,tf.zeros((DIM,DIM-x2-CUTOUT_LENGTH))],axis=1)
                    cutout = tf.reshape(1-filter_,(DIM,DIM,1))
                    image = cutout*image        
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_jpeg_quality(image, 98, 100)

        image = random_rot_shear(image, rot_limit=15, shear_limit=0,)
        image = tf.image.random_hue(image, 0.01)
        image = tf.image.random_saturation(image, 0.70, 1.30)
        image = tf.image.random_contrast(image, 0.80, 1.20)
        image = tf.image.random_brightness(image, 0.2)
        if tf.random.uniform([]) <= 0.3:
            image = tfa.image.gaussian_filter2d(image)
        # image = gaussain_noise(image, p=0.1)
    else:
        image = image * 255.0
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, dtype=tf.uint8)
        image = distort_image(image, config.augname)
        image = tf.cast(image, dtype=tf.float32) / 255.0
    return posting_id, image, label_group, matches

def decode_image(image_data, box, config):
    # image = tf.image.decode_jpeg(image_data, channels = 3)
    # expand_ratio = tf.cast(0.1, tf.float32)
    if box is not None and box[0] != -1:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        # width, height = tf.cast(right - left, tf.float32), tf.cast(bottom - top, tf.float32)
        # h_offset, w_offet = tf.cast(height * expand_ratio, tf.int32), tf.cast(width * expand_ratio, tf.int32)
        # left, top = left - w_offet, top - h_offset
        # right, bottom = right + w_offet, bottom + h_offset
        bbs = tf.convert_to_tensor([top, left, bottom - top, right - left])
        image = tf.io.decode_and_crop_jpeg(image_data, bbs, channels=3)
    else:
        image = tf.image.decode_jpeg(image_data, channels = 3)    

    img_size = config.IMAGE_SIZE
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_image_expand(image_data, box, config, is_train):
    if is_train:
        expand_ratio = tf.random.uniform([], 0.0, 0.2)
    else:
        expand_ratio = tf.constant(0.1, dtype=tf.float32)
    if box is not None and box[0] != -1:
        image = tf.image.decode_jpeg(image_data, channels = 3)    
        shape = tf.shape(image)
        # left, top, right, bottom = box[0], box[1], box[2], box[3]
        left, top, right, bottom = box[3], box[2], box[1], box[0]
        width, height = tf.cast(right - left, tf.float32), tf.cast(bottom - top, tf.float32)
        h_offset, w_offset = height * expand_ratio, width * expand_ratio
        # Make square
        if is_train and tf.random.uniform([]) <= 0.2:
            h_offset += (width - height) / 2
        h_offset, w_offset = tf.cast(h_offset, tf.int32), tf.cast(w_offset, tf.int32)
        left, top = tf.maximum(left - w_offset, 0), tf.maximum(top - h_offset, 0)
        right, bottom = tf.minimum(right + w_offset, shape[1]), tf.minimum(bottom + h_offset, shape[0])
        # bbs = tf.convert_to_tensor([top, left, bottom - top, right - left])
        #image = tf.io.decode_and_crop_jpeg(image_data, bbs, channels=3)
        image = tf.cond(bottom > top, lambda: tf.image.crop_to_bounding_box(image, top, left, bottom - top, right - left), lambda: image)
    else:
        image = tf.image.decode_jpeg(image_data, channels = 3)   

    img_size = config.IMAGE_SIZE
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def read_labeled_tfrecord(config, is_train, example):
    LABELED_TFREC_FORMAT = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "detic_box": tf.io.FixedLenFeature([4], tf.int64),
        "yolov5_box": tf.io.FixedLenFeature([4], tf.int64),
        "backfin_box": tf.io.FixedLenFeature([4], tf.int64),
        # "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['image_name']

    if config.crop_method == 'random':
        if is_train:
            r = tf.random.uniform([])
            bb = tf.cond(r <= 0.4,
                        lambda: tf.cast(example['backfin_box'], tf.int32),
                        lambda: tf.cond(r <= 0.7,
                                       lambda: tf.cast(example['yolov5_box'], tf.int32),
                                       lambda: tf.cast(example['detic_box'], tf.int32)))
            
        else:
            bb = tf.cast(example['detic_box'], tf.int32)
    else:
        bb = tf.cast(example[config.crop_method], tf.int32)

    if config.expand_box:
        image = decode_image_expand(example['image'], bb, config, is_train)
    else:
        image = decode_image(example['image'], bb, config)
    # label_group = tf.one_hot(tf.cast(example['label_group'], tf.int32), depth = N_CLASSES)
    label_group = tf.cast(example['target'], tf.int32)
    # matches = tf.cast(example['species'], tf.int32)
    # matches = example['matches']
    matches = 1
    return posting_id, image, label_group, matches

# This function loads TF Records and parse them into tensors
def load_dataset(filenames, config, ordered=False, is_train=False):
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False 
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
#     dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(partial(read_labeled_tfrecord, config, is_train), num_parallel_calls = AUTO) 
    return dataset

def get_training_dataset(filenames, config):
    dataset = load_dataset(filenames, config, ordered=False, is_train=True)
    dataset = dataset.map(partial(data_augment, config), num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_val_dataset(filenames, config):
    dataset = load_dataset(filenames, config, ordered = True)
    dataset = dataset.map(arcface_format, num_parallel_calls = AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_eval_dataset(filenames, config, get_targets=True):
    dataset = load_dataset(filenames, config, ordered = True)
    dataset = dataset.map(arcface_eval_format, num_parallel_calls = AUTO)
    if not get_targets:
        dataset = dataset.map(lambda image, target: image)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(filenames, config, get_names=True):
    dataset = load_dataset(filenames, config, ordered = True)
    dataset = dataset.map(arcface_inference_format, num_parallel_calls = AUTO)
    if not get_names:
        dataset = dataset.map(lambda image, posting_id: image)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset