import torch
import os
from timeit import default_timer as timer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from unet_class import UNet3D
import torch.optim as optim
import torchvision.transforms as transforms
import train as tr
import transforms as t
import datasets_class as dt
import numpy as np
import matplotlib.pyplot as plt

# Specify the paths to the source and target image stack directories
source_dir = 'data/training_data/source_128/'
target_dir = 'data/training_data/target_128/'
model_dir = 'centroids_v1_128'
model_path = os.path.join(model_dir, "model"+".pth")

# Create the output directory if it does not already exist
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Set validation split
val_split = 0.2

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Set CUDA_LAUNCH_BLOCKING to 1
#torch.backends.cudnn.benchmark = False # Disable cudnn.benchmark


# Define the transforms to apply to the images
train_transform = transforms.Compose([
    # You can add additional transforms here, such as resizing, rotation, etc.
    transforms.ToTensor(),
    t.ImageStackNormalizationTransform()
    #t.RandomCrop3D(output_size=(128, 128, 128)),  # adjust crop size as needed
    #t.Crop3D(output_size=(128, 128, 128))
])

#=====================================================================================================#

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Create a dataset (containing both source and target images)
    dataset = dt.StackDataset(source_dir, target_dir, transform=train_transform)
    print(f'Dataset size: {len(dataset)}')

    # Use GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    # Define the lengths of training and validation sets
    val_size = int(val_split * len(dataset))  # val_split % for validation
    train_size = len(dataset) - val_size      # Remaining (1 - val_split) % for training

    # Use random_split to create training and validation splits and set batch_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = 10

    # Create dataloaders for datasets for both training and validation
    if len(train_dataset) == 0:
        print("Train dataset is empty!")
    elif len(train_dataset) < batch_size:
        print("Train dataset has fewer elements than batch size!")
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    if len(val_dataset) == 0:
        print("Validation dataset is empty!")
    elif len(val_dataset) < batch_size:
        print("Validation dataset has fewer elements than batch size!")
    else:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=True)

    # Before CNN definition, let's check the sizing of input tensor
    data, label = next(iter(train_dataloader))
    print(data.size())
    print(label.size())
    data, label = next(iter(val_dataloader))
    print(data.size())
    print(label.size())

    # Define model
    model = UNet3D().to(device)
    print(model)

    # Specify a loss function and an optimizer to be used for training
    epochs = 2
    learning_rate = 0.001
    # Define class weights
    class_weights = torch.tensor([10.0]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    epoch_number = 0
    train_accuracies_per_epoch = []  # cumulated accuracies from training dataset for each epoch
    val_accuracies_per_epoch = []  # cumulated accuracies from validation dataset for each epoch

    start = timer()
    for t in range(epochs):
        train_accuracy_epoch, val_accuracy_epoch = tr.train_one_epoch(epoch_number, epochs, train_dataloader, val_dataloader, model, loss_fn, optimizer, device)
        train_accuracies_per_epoch.append(train_accuracy_epoch)
        val_accuracies_per_epoch.append(val_accuracy_epoch)
        #test(test_dataloader, model, loss_fn)
        epoch_number += 1
    train_accuracy = np.mean(train_accuracies_per_epoch)
    val_accuracy = np.mean(val_accuracies_per_epoch)
    stop = timer()
    torch.save(model.state_dict(), model_path)
    print("Done!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
