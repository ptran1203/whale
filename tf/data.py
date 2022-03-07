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
import tensorflow_hub as tfhub
from datetime import datetime
from functools import partial

AUTO = tf.data.experimental.AUTOTUNE


def arcface_format(posting_id, image, label_group, matches):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group, matches

def arcface_inference_format(posting_id, image, label_group, matches):
    return image,posting_id

def arcface_eval_format(posting_id, image, label_group, matches):
    return image,label_group

def random_rot_shear(img, rot_limit=10, shear_limit=10):
    rot_d = tf.random.uniform([], -rot_limit, rot_limit)
    shear_d = tf.random.uniform([], -shear_limit, shear_limit)

    if tf.random.uniform([]) <= 0.2:
        img = tfa.image.rotate(img, rot_d * np.pi/180)
    if tf.random.uniform([]) <= 0.2:
        img = tfa.image.shear_x(img, shear_d * np.pi/180, 0.0)
    return img

def random_blur(img, p=0.3, size=3, mean=0.0, std=0.1):
    """Makes 2D gaussian Kernel for convolution."""
    if tf.random.uniform([]) <= p:
        d = tf.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
        gauss_kernel = tf.einsum('i,j->ij',
                                    vals,
                                    vals)

        kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        kernel = kernel[:, :, tf.newaxis, tf.newaxis]
        return tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding="SAME")
    else:
        return img


# Data augmentation function
def data_augment(config, posting_id, image, label_group, matches):

    if config.random_crop:
        image = tf.image.random_crop(image, size=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))
    ### CUTOUT
    if config.CUTOUT and tf.random.uniform([])>0.5:
      N_CUTOUT = 4
      for cutouts in range(N_CUTOUT):
        if tf.random.uniform([])>0.5:
           DIM = config.IMAGE_SIZE
           CUTOUT_LENGTH = DIM//8
           x1 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
           x2 = tf.cast( tf.random.uniform([],0,DIM-CUTOUT_LENGTH),tf.int32)
           filter_ = tf.concat([tf.zeros((x1,CUTOUT_LENGTH)),tf.ones((CUTOUT_LENGTH,CUTOUT_LENGTH)),tf.zeros((DIM-x1-CUTOUT_LENGTH,CUTOUT_LENGTH))],axis=0)
           filter_ = tf.concat([tf.zeros((DIM,x2)),filter_,tf.zeros((DIM,DIM-x2-CUTOUT_LENGTH))],axis=1)
           cutout = tf.reshape(1-filter_,(DIM,DIM,1))
           image = cutout*image

    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_jpeg_quality(image, 90, 100)
    image = random_rot_shear(image, rot_limit=10, shear_limit=10,)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.1)
    image = random_blur(image)
    return posting_id, image, label_group, matches

def decode_image_crop(image_data, box, config):
    # image = tf.image.decode_jpeg(image_data, channels = 3)
    if box is not None and box[0] != -1:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        bbs = tf.convert_to_tensor([top, left, bottom - top, right - left])
        image = tf.io.decode_and_crop_jpeg(image_data, bbs, channels=3)
    else:
        image = tf.image.decode_jpeg(image_data, channels = 3)

    img_size = config.IMAGE_SIZE
    if config.random_crop:
        img_size = int(img_size * 1.15)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_image(image_data, box, config):
    # image = tf.image.decode_jpeg(image_data, channels = 3)
    if box is not None and box[0] != -1:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        bbs = tf.convert_to_tensor([top, left, bottom - top, right - left])
        image = tf.io.decode_and_crop_jpeg(image_data, bbs, channels=3)
    else:
        image = tf.image.decode_jpeg(image_data, channels = 3)    

    img_size = config.IMAGE_SIZE
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def read_labeled_tfrecord(config, example):
    LABELED_TFREC_FORMAT = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "detic_box": tf.io.FixedLenFeature([4], tf.int64),
        # "species": tf.io.FixedLenFeature([], tf.int64),
        # "matches": tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['image_name']
    bb = tf.cast(example['detic_box'], tf.int32)
    image = decode_image(example['image'], bb, config)
    # label_group = tf.one_hot(tf.cast(example['label_group'], tf.int32), depth = N_CLASSES)
    label_group = tf.cast(example['target'], tf.int32)
    # matches = tf.cast(example['species'], tf.int32)
    # matches = example['matches']
    matches = 1
    return posting_id, image, label_group, matches

def read_labeled_tfrecord_train(config, example):
    LABELED_TFREC_FORMAT = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "detic_box": tf.io.FixedLenFeature([4], tf.int64),
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    posting_id = example['image_name']
    bb = tf.cast(example['detic_box'], tf.int32)
    image = decode_image_crop(example['image'], bb, config)
    label_group = tf.cast(example['target'], tf.int32)
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
    if is_train:
        dataset = dataset.map(partial(read_labeled_tfrecord_train, config), num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(partial(read_labeled_tfrecord, config), num_parallel_calls = AUTO) 
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