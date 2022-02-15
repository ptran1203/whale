import albumentations as A

def train_transform(img_size):
    return A.Compose([
        # A.SmallestMaxSize(img_size),
        # A.RandomCrop(img_size, img_size, p=1.0),
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        # A.ImageCompression(quality_lower=99, quality_upper=100, p=0.2),
        # A.GaussNoise(p=0.1),
        # A.GaussianBlur(blur_limit=3, p=0.1),
        # A.RGBShift(),
        # A.RandomBrightnessContrast(p=0.7, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        # A.HueSaturationValue(p=0.7, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        # A.ShiftScaleRotate(p=0.7, shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0),
        # A.Cutout(p=0.5, num_holes=1, max_h_size=int(img_size * 0.3), max_w_size=int(img_size * 0.3)),
        # A.ToGray(p=0.01),
        A.Normalize()
    ])

def val_transform(img_size):
    return A.Compose([
        # A.SmallestMaxSize(img_size),
        # A.CenterCrop(img_size, img_size),
        A.Resize(img_size, img_size),
        A.Normalize()
    ])
