import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *


transform = transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # 讀 SegmentationClass（.png）→ 去掉副檔名
        mask_names = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(path, 'SegmentationClass.256'))]
        image_dir = os.path.join(path, 'JPEGImages.256')
        available_images = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir) if f.endswith('.jpg')}

        self.name_pairs = []
        for name in mask_names:
            if name in available_images:
                image_file = available_images[name]
                mask_file = name + '.png'
                self.name_pairs.append((image_file, mask_file))

    def __len__(self):
        return len(self.name_pairs)

    def __getitem__(self, index):
        image_name, mask_name = self.name_pairs[index]
        image_path = os.path.join(self.path, 'JPEGImages.256', image_name)
        segment_path = os.path.join(self.path, 'SegmentationClass.256', mask_name)

        image = keep_image_size_open_rgb(image_path, size=(256, 256))
        segment_image = keep_image_size_open(segment_path, size=(256, 256))


        return transform(image), torch.Tensor(np.array(segment_image))

if __name__ == '__main__':
    from torch.nn.functional import one_hot
    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out = one_hot(data[0][1].long())
    print(out.shape)