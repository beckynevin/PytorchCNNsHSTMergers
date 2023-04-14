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
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.load(img_path) #keep as np array to normalize
        image = image[:,:,1:4]
        #print('image shape: ', np.shape(image))
        image = stretch(image)
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
        T.Pad(11)
        ))
    else: transforms.append(T.Pad(11))
        
    return T.Compose(transforms)

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

accuracy = np.array([])

train_mergers_dataset = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(train=True))
train_nonmergers_dataset = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(train=True))

train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset, train_nonmergers_dataset])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 1, batch_size=32)

validation_mergers_dataset = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(train=False))
validation_nonmergers_dataset = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(train=False))

validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset, validation_nonmergers_dataset])
validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = 1, batch_size=32)#num workers used to be 4

#images, labels = next(iter(train_dataloader)) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = models.alexnet(pretrained=True, progress = True)
print(model)

#tweak model - changing layers didnt work so lets get rid of it lol 
# model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=(5,5), stride=(4, 4), padding=(2, 2))
# model.classifier[0] = torch.nn.Dropout(p=0.3, inplace=False)
# model.classifier[1] = torch.nn.Linear(in_features=512, out_features=256, bias=True)
# model.classifier[2] = torch.nn.ReLU(inplace=True)
# model.classifier[3] = torch.nn.Dropout(p=0.3, inplace=False)
# model.classifier[4] = torch.nn.Linear(in_features=256, out_features=128, bias=True)
# model.classifier[5] = torch.nn.ReLU(inplace=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
print(model)

NUM_EPOCHS = 5
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0

optimizer = optim.AdamW(model.parameters(), lr=0.0001)

for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_dataloader):
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        #print(images.size())
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.unsqueeze(1)
        #print(outputs.size())
        #print(labels.size())
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
    
    val_error_count = 0.0
    for images, labels in iter(validation_dataloader):
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        outputs = model(images)
        val_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    val_accuracy = 1.0 - float(val_error_count) / float(len(validation_dataset_full))
    print('%d: %f' % (epoch, val_accuracy))
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = val_accuracy
    np.append(accuracy, val_accuracy)
    save_checkpoint(model=model, optimizer=optimizer, save_path='/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/savedmodel.txt', epoch = epoch)
print('best accuracy:', best_accuracy)
np.savetxt('/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/accuracy_transfer.txt', accuracy)
