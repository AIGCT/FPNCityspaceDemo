import os
import random
import shutil


def makeLabel():
    # 数据集路径
    dataset_path = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/getFine_trainvaltest/gtFine"

    #原始的train, valid文件夹路径
    train_dataset_path = os.path.join(dataset_path, 'train')
    val_dataset_path = os.path.join(dataset_path, 'val')
    test_dataset_path = os.path.join(dataset_path, 'test')

    # 创建train,valid的文件夹
    train_path = '/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/trainImg'

    train_images_path = os.path.join(train_path, 'classes_train')
    val_images_path = os.path.join(train_path, 'classes_val')
    test_images_path = os.path.join(train_path, 'classes_test')

    if os.path.exists(train_path) == False:
        os.mkdir(train_path)
    if os.path.exists(train_images_path) == False:
        os.mkdir(train_images_path)
    if os.path.exists(val_images_path) == False:
        os.mkdir(val_images_path)
    if os.path.exists(test_images_path) == False:
        os.mkdir(test_images_path)

    # -----------------移动文件---对于19类语义分割, 主需要原始图像中的labelIds结尾图片-----------------------
    for file_name in os.listdir(train_dataset_path):
        file_path = os.path.join(train_dataset_path, file_name)
        for image in os.listdir(file_path):
            # 查找对应的后缀名，然后保存到文件中
            if image.split('.png')[0][-13:] == "labelTrainIds":
                # print(image)
                shutil.copy(os.path.join(file_path, image),
                            os.path.join(train_images_path, image))

    for file_name in os.listdir(val_dataset_path):
        file_path = os.path.join(val_dataset_path, file_name)
        for image in os.listdir(file_path):
            if image.split('.png')[0][-13:] == "labelTrainIds":
                shutil.copy(os.path.join(file_path, image),
                            os.path.join(val_images_path, image))

    for file_name in os.listdir(test_dataset_path):
        file_path = os.path.join(test_dataset_path, file_name)
        for image in os.listdir(file_path):
            if image.split('.png')[0][-13:] == "labelTrainIds":
                shutil.copy(os.path.join(file_path, image),
                            os.path.join(test_images_path, image))


if __name__ == '__main__':
    makeLabel()
