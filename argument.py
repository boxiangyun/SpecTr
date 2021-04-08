import cv2
from skimage.transform import AffineTransform, warp
import numpy as np
import skimage.io
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as AF

def resize(image, size=(128, 128)):
    return cv2.resize(image, size)


def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image, mask=None):
    if mask is None:
        return aug(image=image)['image']
    else:
        augment = aug(image=image,mask=mask)
        return augment['image'],augment['mask']


class Transform:
    def __init__(self,  size=None, train=True,
                 BrightContrast_ration=0.,  noise_ratio=0., cutout_ratio=0.,
                 grid_distortion_ratio=0., elastic_distortion_ratio=0., 
                 piece_affine_ratio=0., ssr_ratio=0.,Rotate_ratio=0.,Flip_ratio=0.):
        self.size = size
        self.train = train
        self.noise_ratio = noise_ratio
        self.BrightContrast_ration = BrightContrast_ration
        self.cutout_ratio = cutout_ratio
        self.grid_distortion_ratio = grid_distortion_ratio
        self.elastic_distortion_ratio = elastic_distortion_ratio
        self.piece_affine_ratio = piece_affine_ratio
        self.ssr_ratio = ssr_ratio
        self.Rotate_ratio = Rotate_ratio
        self.Flip_ratio = Flip_ratio

        
    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example
        # --- Augmentation ---
        # --- Train/Test common preprocessing ---

        if self.size is not None:
            x = resize(x, size=self.size)

        # albumentations...
        
        # # 1. blur
        if _evaluate_ratio(self.BrightContrast_ration):
                x = apply_aug(A.RandomBrightnessContrast(p=1.0), x)
        #
        if _evaluate_ratio(self.noise_ratio):
            r = np.random.uniform()
            if r < 0.50:
                x = apply_aug(A.GaussNoise(var_limit=5. / 255., p=1.0), x)
            else:
                x = apply_aug(A.MultiplicativeNoise(p=1.0), x)
        
        # if _evaluate_ratio(self.cutout_ratio):
        #     # A.Cutout(num_holes=2,  max_h_size=2, max_w_size=2, p=1.0)  # Deprecated...
        #     x = apply_aug(A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=1.0), x)
        #
        if _evaluate_ratio(self.grid_distortion_ratio):
             x,y = apply_aug(A.GridDistortion(p=1.0), x,y)
        #
        if _evaluate_ratio(self.elastic_distortion_ratio):
             x,y = apply_aug(A.ElasticTransform(
                 sigma=50, alpha=1,  p=1.0), x,y)
        #
        if _evaluate_ratio(self.Rotate_ratio):
            x,y = apply_aug(A.Rotate(p=1.0),x,y)
        
        if _evaluate_ratio(self.Flip_ratio):
            x,y = apply_aug(A.Flip(p=1.0),x,y)
            
            
            
        # if _evaluate_ratio(self.piece_affine_ratio):
        #     x = apply_aug(A.IAAPiecewiseAffine(p=1.0), x)
        #
#         if _evaluate_ratio(self.ssr_ratio):
#             x = apply_aug(A.ShiftScaleRotate(
#                 shift_limit=0.0625,
#                 scale_limit=0.1,
#                 rotate_limit=30,
#                 p=1.0), x)
        if self.train:
            #y = y.astype(np.int64)
            return x, y
        else:
            return x
      
if __name__ == '__main__':
      import matplotlib.pyplot as plt
      from tqdm import tqdm 
      f, ax = plt.subplots(3,3, figsize=(16,18))
      img = skimage.io.imread('/home/ubuntu/Pictures/timg.jpeg')
      transform = Transform(affine=False,train=False,Flip_ratio=0.8)
      for i in tqdm(range(9)):
            aug_img = transform(img)
            ax[i//3,i%3].imshow(aug_img)
            #ax[i//3, i%3].set_title(f'aug_method:{infor}')
      plt.show()
      