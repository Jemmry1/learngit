from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self,root_dir)
       

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index] #根据索引 index 获取当前图像文件名。
        img_path = os.path.join(self.root_dir, img_file) #将根目录路径和图像文件名结合成完整的图像路径。
        image = np.array(Image.open(img_path)) #使用 PIL.Image.open 打开图像文件并将其转换为 NumPy 数组。
        input_image = image[:, :600, :] #前600为输入图像
        target_image = image[:, 600:, :] #后600像素为对应的目标图像

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_imaeg, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_input(image=target_image)["image"]

        return input_imaeg, target_image
