import albumentations as A

def train_transform(img_size):
    return A.Compose([
        A.Resize(int(img_size * 1.0), int(img_size * 1.0)),
        # A.RandomCrop(img_size, img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=99, quality_upper=100, p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.RandomBrightnessContrast(p=0.75, brightness_limit=0.1, contrast_limit=0.1),
        A.HueSaturationValue(p=0.75, hue_shift_limit=0.01, sat_shift_limit=0.3, val_shift_limit=0.2),
        A.Cutout(p=0.5, num_holes=1, max_h_size=int(img_size * 0.1), max_w_size=int(img_size * 0.1)),
        A.ShiftScaleRotate(p=0.75, shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=0),
        A.Normalize()
    ])

def val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize()
    ])
