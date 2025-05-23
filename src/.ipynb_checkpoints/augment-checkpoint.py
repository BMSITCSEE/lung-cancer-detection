import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.GaussianBlur(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
