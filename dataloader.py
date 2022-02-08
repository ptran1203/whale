import os
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from glob import glob

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
    A.GaussNoise(p=0.1),
    A.GaussianBlur(blur_limit=3, p=0.1),
    A.RGBShift(),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    A.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
    A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, border_mode=0),
    A.Cutout(p=0.1, num_holes=1, max_h_size=32, max_w_size=32),
    A.Normalize()
])

val_transform = A.Compose([
    A.Normalize()
])

class WhaleDataset(Dataset):
    def __init__(self, train_df, img_dir, img_size=256, transform=train_transform):
        self.df = train_df
        self.transform = transform
        self.img_size = img_size
        self.df['img_path'] = self.df['image'].apply(lambda x: os.path.join(img_dir, x))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path, label = row['img_path'], row['label']
        img = cv2.imread(img_path)
        assert img is not None, img_path
        img = cv2.resize(img[:, :, ::-1], (self.img_size, self.img_size))
        if self.transform is not None:
            img = self.transform(image=img)['image']

        return torch.from_numpy(img.transpose(2, 0, 1)), torch.tensor(label), row['image']

    def __len__(self):
        return len(self.df)


class InferDataset(Dataset):
    def __init__(self, img_dir, img_size=256, transform=val_transform):
        self.transform = transform
        self.img_size = img_size
        self.load_data(img_dir)

    def __getitem__(self, index):
        img_path = self.data[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img[:, :, ::-1], (self.img_size, self.img_size))
        img = self.transform(image=img)['image']

        return torch.from_numpy(img.transpose(2, 0, 1)), os.path.basename(img_path)

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