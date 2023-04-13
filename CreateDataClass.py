import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
from astropy.visualization import AsinhStretch
import os
from torchvision.io import read_image

path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/z1mocks/'
stretch = AsinhStretch()

class BinaryMergerDataset(Dataset):
    def __init__(self, data_path, dataset, mergers = True, transform=None):
        self.dataset = dataset
        self.mergers = mergers
        if self.dataset == 'train':
            if mergers == True:
                self.images = glob.glob(data_path + 'training/anymergers/allfilters*.npy')
                self.img_labels = glob.glob(data_path + 'training/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'training/nonmergers/allfilters*.npy')
                self.img_labels = glob.glob(data_path + 'training/nonmergers/mergerlabel.npy')
        elif self.dataset == 'validation':
            if mergers == True:
                self.images = glob.glob(data_path + 'validation/anymergers/allfilters*.npy')
                self.img_labels = glob.glob(data_path + 'validation/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'validation/nonmergers/allfilters*.npy')
                self.img_labels = glob.glob(data_path + 'validation/nonmergers/mergerlabel.npy')
        elif self.dataset == 'test':
            if mergers == True:
                self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
                self.img_labels = glob.glob(data_path + 'test/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
                self.img_labels = glob.glob(data_path + 'test/nonmergers/mergerlabel.npy')
        
        self.transform = transform
        

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.load(img_path) #keep as np array to normalize
        image = stretch(image)
        label = self.img_labels[idx]
        if self.transform is not None:
            image = self.transform(image)

        return image, label



# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


def get_transforms(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train == True:
        transforms.append(torch.nn.Sequential(
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        ))
        
    return T.Compose(transforms)

train_mergers_dataset = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(train=True))
train_nonmergers_dataset = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(train=True))

train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset, train_nonmergers_dataset])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 4)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="magma", norm = LogNorm())
plt.show()
print(f"Label: {label}")
