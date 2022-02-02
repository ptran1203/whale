import os
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from glob import glob

class StampDataset(Dataset):

    def __init__(self, img_dir, img_size=256, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_size = img_size
        self.load_dataset()

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = cv2.imread(img_path)
        assert img is not None, img_path
        img = cv2.resize(img[:, :, ::-1], (self.img_size, self.img_size))
        if self.transform is not None:
            img = self.transform(image=img)['image']

        return torch.from_numpy(img.transpose(2, 0 , 1)), torch.tensor(label)

    def __len__(self):
        return len(self.data)

    def load_dataset(self):
        img_dir = self.img_dir
        all_classes = os.listdir(img_dir)
        skip_classes = ['Other', '295-83']
        all_classes = [x for x in all_classes if x not in skip_classes]

        self.data = []
        self.n_classes = len(all_classes)
        self.le = LabelEncoder()
        self.le.fit(all_classes)
        for c in all_classes:
            path = os.path.join(img_dir, c)
            imgs = glob(os.path.join(path, "*"))
            y_trans = self.le.transform([c])
            for img in imgs:
                self.data.append((img, y_trans[0]))
        
        print(f"Total images: {len(self.data)}, {len(all_classes)} classes")


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

if __name__ == '__main__':
    dataset = StampDataset('/opt/data/vu/stamp_comp/Sorted_data1')

    for img, c in dataset:
        print(img.shape, c)
        break