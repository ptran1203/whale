"""
Convert  Yolov5 output to annotation csv
"""
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

_dir = '../ann_whale_det'

size_df = pd.concat([pd.read_csv('data/train_image_size.csv'), pd.read_csv('data/test_image_size.csv')])

# print(size_df)

img2w = dict(zip(size_df['image'], size_df['w']))
img2h = dict(zip(size_df['image'], size_df['h']))


def draw_box(img, box):
    pt1, pt2 = (box[0], box[1]), (box[0] + box[2], box[1] + box[3])
    return cv2.rectangle(img, pt1, pt2, (255, 0 ,0), thickness=2)

def decode_ann(line, img_w, img_h):
    bb = [float(x) for x in line.strip().split(' ')]
    xc, yc, w, h = bb[1:5]
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x, y = xc - w // 2, yc - h // 2
    return int(x), int(y), int(w) ,int(h)

anns = []
files = glob(os.path.join(_dir, 'train/labels/*')) + glob(os.path.join(_dir, 'test/labels/*')) 
# files = [np.random.choice(files)]
for fi in tqdm(files):
    with open(fi, 'r') as f:
        is_train = 'train/labels' in fi
        lines = f.read().split('\n')
        img_id = os.path.basename(fi).split('.')[0] + '.jpg'
        img_w, img_h = img2w[img_id], img2h[img_id]
        x, y, w, h = decode_ann(lines[0], img_w, img_h)
        anns.append([img_id, img_w, img_h, x, y, w, h, 'train' if is_train else 'test'])

ann_df = pd.DataFrame(anns, columns=['image', 'img_w', 'img_h', 'x', 'y', 'w', 'h', 'subset'])
ann_df.to_csv('data/ann.csv', index=False)

# img = cv2.imread(f'../train_images-384-384/{img_id}')[:,:,::-1]
# assert img is not None
# img_w, img_h = img.shape[1], img.shape[0]
# img = draw_box(img, box)
# img = cv2.UMat.get(img)
# plt.imshow(img)
# plt.show()