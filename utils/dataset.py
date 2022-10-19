# 导入库
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import DataLoader
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.filterwarnings("ignore")
torch.manual_seed(17)

# 设置数据集路径
x_train_dir = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/trainImg/train"
y_train_dir = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/trainImg/classes_train"

x_valid_dir = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/trainImg/val"
y_valid_dir = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/trainImg/classes_val"


class CityScapesDataset(torch.utils.data.Dataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """

    def __init__(self, images_dir, masks_dir):
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(),
            # A.RandomBrightnessContrast(),
            A.RandomSnow(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.ids = os.listdir(images_dir)
        self.ids2 = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id)
                           for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id)
                          for image_id in self.ids2]

    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('RGB'))
        image = self.transform(image=image, mask=mask)

        return image['image'], image['mask'][:, :, 0]

    def __len__(self):
        return len(self.ids)


def loader(batch_size=128):

    # start
    print("start loading data")

    bs = batch_size

    train_dataset = CityScapesDataset(
        x_train_dir,
        y_train_dir,
    )
    val_dataset = CityScapesDataset(
        x_valid_dir,
        y_valid_dir,
    )

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    # end
    print("end loading data")

    return train_loader, val_loader


if __name__ == "__main__":

    train_loader, val_loader = loader()

    # test data
    # for index, (img, label) in enumerate(train_loader):
    #     print(img.shape)
    #     print(label.shape)

    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(221)
    #     plt.imshow((img[0, :, :, :].moveaxis(0, 2)))
    #     plt.subplot(222)
    #     plt.imshow(label[0, :, :])

    #     plt.subplot(223)
    #     plt.imshow((img[6, :, :, :].moveaxis(0, 2)))
    #     plt.subplot(224)
    #     plt.imshow(label[6, :, :])
    #     plt.show()
    #     if index == 0:
    #         break
