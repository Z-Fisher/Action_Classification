import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# Load Model
network_path = 'models/fused_epoch6'
if torch.cuda.is_available():
    checkpoint = torch.load(network_path)
else:
    checkpoint = torch.load(network_path,  map_location=torch.device('cpu'))

# Load Saved Data
num_epochs = checkpoint['epoch']
train_total_loss_list = checkpoint['train_total_loss_list']
epoch_total_loss_list = checkpoint['epoch_total_loss_list']
test_loss_list = checkpoint['test_loss_list']
train_counter = checkpoint['train_counter']
accuracy_list = checkpoint['accuracy_list']

epoch_list = np.arange(num_epochs+1)

# Training Loss
fig = plt.figure()
plt.plot(epoch_list, epoch_total_loss_list, color='blue')
plt.legend(['FuseNet Train Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Total Loss')

# Test Loss
fig = plt.figure()
plt.plot(epoch_list, test_loss_list, color='red')
plt.legend(['Validation Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Total Loss')

# Accuracy
fig = plt.figure()
plt.plot(epoch_list, accuracy_list, color='red')
plt.legend(['FuseNet Validation Accuracy'], loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()