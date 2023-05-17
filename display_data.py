import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import glob
from astropy.visualization import AsinhStretch
#import os
#from torchvision.io import read_image
from tqdm import tqdm

class BinaryMergerDataset(Dataset): #in future: put this in one file and always call it!
    def __init__(self, data_path, dataset, mergers = True, transform=None, codetest=False):
        self.dataset = dataset
        self.mergers = mergers     
        self.codetest=codetest
        if self.dataset == 'train':
            if mergers == True:
                self.images = glob.glob(data_path + 'training/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'training/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'training/nonmergers/mergerlabel.npy')
        elif self.dataset == 'validation':
            if mergers == True:
                self.images = glob.glob(data_path + 'validation/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'validation/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'validation/nonmergers/mergerlabel.npy')
        elif self.dataset == 'test':
            if mergers == True:
                self.images = glob.glob(data_path + 'test/anymergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/anymergers/mergerlabel.npy')
            else:
                self.images = glob.glob(data_path + 'test/nonmergers/allfilters*.npy')
                self.img_labels = np.load(data_path + 'test/nonmergers/mergerlabel.npy')
        
        self.transform = transform
        

    def __len__(self):
        if self.codetest:
            return len(self.img_labels[0:10])
        else:   
            return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.load(img_path) #keep as np array to normalize
        image = image[:,:,1:4]
        #print('image shape: ', np.shape(image))
        #image = stretch(image)
        label_file = self.img_labels
        #print('first label call: ', np.shape(label))
        label = label_file[idx]
        # if label != 0:
        #     print(label)
        #print(labels)
        #label = np.load(label_path)[idx]
        #print('label shape: ',np.shape(labels))
        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)



def get_transforms(pad_val, train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train == True:
        transforms.append(torch.nn.Sequential(
        T.RandomRotation(30), 
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.Pad(pad_val)
        ))
    else: transforms.append(T.Pad(pad_val))
        
    return T.Compose(transforms)


def main():
    path = 'data/'
    stretch = AsinhStretch()
    pad_val = int((256-202)/2)

    print('loading data')

    train_mergers_dataset = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(pad_val, train=True), codetest=True)
    train_nonmergers_dataset = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(pad_val, train=True), codetest=True)

    train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset, train_nonmergers_dataset])
    train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 1, batch_size=32)

    for batch in train_dataloader:
        images, labels = batch
        # Assuming the images are in tensor format, convert them to numpy arrays
        images = images.numpy()
        labels = labels.numpy()
        break  # Break after the first batch to plot a few images
    
    print(np.shape(images), 'shape of images')

    fig, axs = plt.subplots(4, 8, figsize=(12, 6))
    axs = axs.flatten()

    for i in range(len(axs)):
        image = np.transpose(images[i], (1, 2, 0))  # Transpose image dimensions if necessary
        label = labels[i]
        
        axs[i].imshow(image)
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!