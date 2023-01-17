import torchvision
import torch

# a simple function to load tiny-imagenet dataset
# put this file and the 'tiny-imagenet-200' folder under the same path
size = 112
train_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(size, size)),
     torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/train', transform=train_transform)
val_dataset = torchvision.datasets.ImageFolder(root='./tiny-imagenet-200/val', transform=train_transform)
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print('Successfully load data!')
