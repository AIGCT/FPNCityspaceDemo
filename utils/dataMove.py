import os
import random
import shutil


def makeDataSet():
    # 数据集路径
    dataset_path = "/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/leftImg8bit"

    #原始的train, valid文件夹路径
    train_dataset_path = os.path.join(dataset_path, 'train')
    val_dataset_path = os.path.join(dataset_path, 'val')
    test_dataset_path = os.path.join(dataset_path, 'test')

    # 创建train,valid的文件夹
    train_path = '/home/yangdenghui/nfs/yangdenghui/SemanticSegmentationUsingPFPN/Cityscapes/trainImg'
    train_images_path = os.path.join(train_path, 'train')
    val_images_path = os.path.join(train_path, 'val')
    test_images_path = os.path.join(train_path, 'test')

    if os.path.exists(train_path) == False:
        os.mkdir(train_path)
    if os.path.exists(train_images_path) == False:
        os.mkdir(train_images_path)
    if os.path.exists(val_images_path) == False:
        os.mkdir(val_images_path)
    if os.path.exists(test_images_path) == False:
        os.mkdir(test_images_path)

    # -----------------移动文件夹-------------------------------------------------
    print("移动训练集文件夹")
    for file_name in os.listdir(train_dataset_path):
        file_path = os.path.join(train_dataset_path, file_name)
        for image in os.listdir(file_path):
            shutil.copy(os.path.join(file_path, image),
                        os.path.join(train_images_path, image))

    print("移动验证集文件夹")
    for file_name in os.listdir(val_dataset_path):
        file_path = os.path.join(val_dataset_path, file_name)
        for image in os.listdir(file_path):
            shutil.copy(os.path.join(file_path, image),
                        os.path.join(val_images_path, image))

    print("移动测试集文件夹")
    for file_name in os.listdir(test_dataset_path):
        file_path = os.path.join(test_dataset_path, file_name)
        for image in os.listdir(file_path):
            shutil.copy(os.path.join(file_path, image),
                        os.path.join(test_images_path, image))


if __name__ == '__main__':
    makeDataSet()
