import albumentations as A

def train_transform(img_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75, brightness_limit=0.1, contrast_limit=0.1),
        A.HueSaturationValue(p=0.75, hue_shift_limit=0.01, sat_shift_limit=0.3, val_shift_limit=0.2),
        A.CLAHE(p=0.1),
        A.MotionBlur(p=0.2),
        A.Resize(img_size, img_size),
        # A.Cutout(p=0.0, num_holes=1, max_h_size=int(img_size * 0.1), max_w_size=int(img_size * 0.1)),
        A.Normalize()
    ])

def val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize()
    ])
