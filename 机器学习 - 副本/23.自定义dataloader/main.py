
import matplotlib.pyplot as plt
import numpy as np
def load_annotations(ann_file):
    data_infos ={}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for filename, gt_label in samples:
            data_infos[filename]= np.array(gt_label, dtype=np.int64)
    return data_infos

a = load_annotations('txt.txt')
print(a)
x = list(a.keys())
label = list(a.values())

import os
import torch
data_dir = 'E:\gupao\项目实战\机器学习/'
train_dir = data_dir + '/23.'
image_path = [os.path.join(train_dir, img) for img in x]


from torch.utils import Dataset, DataLoader
class FowerDataset(Daraset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.key())]
        self.label = [label for label in list(self.img_label.value())]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = Image. open(self.img[idx])
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy (np.array(label))
        return image, label








































print()







