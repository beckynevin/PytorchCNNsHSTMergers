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



# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


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

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

accuracylist = []

train_mergers_dataset = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(train=True), codetest=True)
train_nonmergers_dataset = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(train=True), codetest=True)

train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset, train_nonmergers_dataset])
train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 1, batch_size=BATCH_SIZE)

validation_mergers_dataset = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(train=False), codetest=True)
validation_nonmergers_dataset = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(train=False), codetest=True)

validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset, validation_nonmergers_dataset])
validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = 1, batch_size=BATCH_SIZE)#num workers used to be 4

#images, labels = next(iter(train_dataloader)) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#common practice is to make a subclass
class ResNet(nn.Module): #inheritance --> can use anything in nn.Module NOT LIKE A FUNCTION
    def __init__(
        self, in_channels: int,  out_channels: int, pretrained: bool = True 
    ):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #self says "this variable belongs to the class"
        #print(self.resnet)
        # Freeze model parameters -- commented out on 5/11/23 to test if that's whats messing with accuracy
        for param in self.resnet.parameters():
            param.requires_grad = False
        #self.resnet.fc = nn.Linear(in_channels, out_channels, bias=True) #bias is like y-intercept #add activation here
        self.resnet.fc = nn.Sequential(torch.nn.Linear(in_channels, out_channels, bias=True), torch.nn.Sigmoid())
        #print(self.resnet)

    def forward(self, x): #how a datum moves through the net
        x = self.resnet(x) #model already has the sequence - propogate x through the network!
        print(x)
        return x

def main():
    path = 'data/'
    stretch = AsinhStretch()
    pad_val = int((256-202)/2)

    accuracylist = []

    train_mergers_dataset = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(pad_val, train=True), codetest=True)
    train_nonmergers_dataset = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(pad_val, train=True), codetest=True)

    train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset, train_nonmergers_dataset])
    train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = 1, batch_size=32)

    validation_mergers_dataset = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(pad_val, train=False), codetest=True)
    validation_nonmergers_dataset = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(pad_val, train=False), codetest=True)

    validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset, validation_nonmergers_dataset])
    validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = 1, batch_size=32)#num workers used to be 4

    #images, labels = next(iter(train_dataloader)) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'device is {device}')

    print('running model')
    model = ResNet(512, 1, True)
    model = model.to(device)
    print(model)

#tweak model
#model.features[0] = torch.nn.Conv2d(model.features[0].kernel_sieze, (5,5))
#model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 1)
print(model)
NUM_EPOCHS = 10
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0
# training_epoch_loss = []
# val_epoch_loss = []
# training_epoch_accuracy = []
# val_epoch_accuracy = []
# accuracy = []
# train_loss = 0.0
# train_acc = 0.0
# valid_loss = 0.0
# valid_acc = 0.0

loss = {} #loss history
loss['train'] = []
loss['validation'] = []
err = {} #used later for accuracy
err['train'] = []
err['validation'] = []
x_epoch = []

fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

optimizer = optim.Adam(model.parameters(), lr=0.001)

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, loss['validation'], 'ro-', label='val')
    ax1.plot(x_epoch, err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, err['validation'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig('metrics.png')

trainingloss = 0.0
correct_labels_train = 0.0
valloss = 0.0
correct_labels_val = 0.0



for epoch in range(NUM_EPOCHS):
    
    #train_error_count = 0.0
    for images, labels in tqdm(iter(train_dataloader)):
        model.train(True) #default is not in training mode - need to tell pytorch to train
        images = images.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        #print(images.size())
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.unsqueeze(1)
        #print(outputs.size())
        #print(labels.size())
        loss = F.binary_cross_entropy(outputs, labels)
        # Calculate Loss
        trainingloss += loss.item() * BATCH_SIZE
        correct_labels_train += float(torch.sum(outputs == labels.data))
        loss.backward()
        optimizer.step()
        #train_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
       # print('length of training loss', len(trainingloss))
        training_epoch_loss = trainingloss / len(train_dataloader.dataset)
        print(type(training_epoch_loss))
        training_epoch_accuracy = correct_labels_train / len(train_dataloader.dataset)
        #loss['train'].append(training_epoch_loss)
        err['train'].append(1.0 - training_epoch_accuracy)        
    # training_epoch_loss.append(trainingloss)
    # print('shape of training loss: ', np.shape(training_epoch_loss))
    # train_accuracy = 1.0 - float(train_error_count) / float(len(train_dataset_full))
    # training_epoch_accuracy.append(train_accuracy)
    
    #val_error_count = 0.0
    for images, labels in iter(validation_dataloader):
        model.train(False)
        model.eval() #added 5/13/23
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        outputs = model(images)
        labels = labels.unsqueeze(1)
        loss = F.binary_cross_entropy(outputs, labels)
        #valloss.append(loss.item())
        #val_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        valloss += loss.item() * BATCH_SIZE
        correct_labels_val += float(torch.sum(outputs == labels.data))
        validation_epoch_loss = valloss / len(validation_dataloader.dataset)
        validation_epoch_accuracy = correct_labels_val / len(validation_dataloader.dataset)
        loss['validation'].append(validation_epoch_loss)
        err['validation'].append(1.0 - validation_epoch_accuracy) 
        
    draw_curve(epoch)
    save_checkpoint(model=model, optimizer=optimizer, save_path='/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/savedresnetmodel.txt', epoch = epoch)

    # val_epoch_loss.append(np.array(valloss))
    # val_accuracy = 1.0 - float(val_error_count) / float(len(validation_dataset_full))
    # accuracylist.append(val_accuracy)
    # print('%d: %f' % (epoch, val_accuracy))
    # if val_accuracy > best_accuracy:
    #     torch.save(model.state_dict(), BEST_MODEL_PATH)
    #     best_accuracy = val_accuracy

def Accuracy(model, dataloader): #https://blog.paperspace.com/training-validation-and-accuracy-in-pytorch/
    """
    This function computes accuracy
    """
    #  setting model state
    model.eval()

    #  instantiating counters
    total_correct = 0
    total_instances = 0

    #  creating dataloader
    #dataloader = DataLoader(dataset, 64)

    #  iterating through batches
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            #-------------------------------------------------------------------------
            #  making classifications and deriving indices of maximum value via argmax
            #-------------------------------------------------------------------------
            classifications = torch.argmax(model(images), dim=1)

            #--------------------------------------------------
            #  comparing indicies of maximum values and labels
            #--------------------------------------------------
            correct_predictions = sum(classifications==labels).item()

            #------------------------
            #  incrementing counters
            #------------------------
            total_correct+=correct_predictions
            total_instances+=len(images)
    return round(total_correct/total_instances, 3)

#print('best accuracy:', best_accuracy)
#training accuracy
training_accuracy = Accuracy(model, train_dataloader)
validation_accuracy = Accuracy(model, validation_dataloader)

accuracylist = np.array(accuracylist)
np.savetxt('/n/home09/aschechter/code/BinaryCNNTesting/PytorchCNNs/accuracy_resnettransfer.txt', accuracylist)


## plot training and validation loss
plt.figure()
plt.plot(training_epoch_loss, label = 'training')
plt.plot(validation_epoch_loss, label = 'validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('ResNet_loss.png')

    plt.figure()
    plt.plot(np.arange(0,NUM_EPOCHS), training_epoch_accuracy, label = 'training')
    plt.plot(np.arange(0,NUM_EPOCHS), accuracylist, label = 'validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('ResNet_accuracy.png')


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!