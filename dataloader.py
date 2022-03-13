import os
import cv2
import numpy as np
import random
import albumentations as A
import torch
import math
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from glob import glob

try:
    import torch_xla
except:
    pass

def _slurp_file(path):
    fstat = torch_xla._XLAC._xla_tffile_stat(path)
    gcs_file = torch_xla._XLAC._xla_tffile_open(path)
    return torch_xla._XLAC._xla_tffile_read(gcs_file, 0, fstat['length'])

def random_perspective(im, degrees=30, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(0, 0, 0))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

    return im

class WhaleDataset(Dataset):
    def __init__(self, train_df, img_dir, img_size=256, transform=None, cv2_aug=False):
        self.df = train_df
        self.transform = transform
        self.img_size = img_size
        self.cv2_aug = cv2_aug
        if 'seg_img' in img_dir:
            # PNG
            self.df['img_path'] = self.df['image'].apply(lambda x: os.path.join(img_dir, x.replace('.jpg', '.png')))
        else:
            self.df['img_path'] = self.df['image'].apply(lambda x: os.path.join(img_dir, x))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path, label = row['img_path'], row['label']
        if img_path.startswith('gs://'):
            bf = _slurp_file(img_path)
            img = cv2.imdecode(np.frombuffer(bf, dtype=np.int8), flags=1)
        else:
            img = cv2.imread(img_path)
        assert img is not None, img_path

        img = img[:,:,::-1]

        if self.cv2_aug:
            img = random_perspective(img, degrees=10, translate=0.0, scale=0.2, shear=5, perspective=0.001)
        if self.transform is not None:
            img = self.transform(image=img)['image']

        return torch.as_tensor(img.transpose(2, 0, 1)), torch.as_tensor(label), row['image']

    def __len__(self):
        return len(self.df)


class InferDataset(Dataset):
    def __init__(self, img_dir, img_size=256, transform=None):
        self.transform = transform
        self.img_size = img_size
        self.load_data(img_dir)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = cv2.imread(img_path)[:, :, ::-1]
        img = self.transform(image=img)['image']

        return torch.as_tensor(img.transpose(2, 0, 1)), os.path.basename(img_path)

    def __len__(self):
        return len(self.data)

    def is_img_file(self, img_file):
        ext = img_file.split(".")[-1]
        return ext in {'jpg', 'png', 'jpeg'}

    def load_data(self, img_dir):
        self.data = glob(os.path.join(img_dir, '*'))
        self.data = [x for x in self.data if self.is_img_file(x)]




if __name__ == '__main__':
    dataset = WhaleDataset('/opt/data/vu/stamp_comp/Sorted_data1')

    for img, c in dataset:
        print(img.shape, c)
        break