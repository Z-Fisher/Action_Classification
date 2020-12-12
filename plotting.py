import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset import *
from SpatialStream import *

# Load Model
network_path = 'models/spatial_epoch11'
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

# Train Loss
# fig = plt.figure()
# plt.plot(train_counter, train_class_loss_list, color='blue')
# plt.legend(['Training Class Loss'], loc='upper right')
# plt.xlabel('Training Iterations')
# plt.ylabel('Class Loss')
#
# fig = plt.figure()
# plt.plot(train_counter, train_reg_loss_list, color='blue')
# plt.legend(['Training Reg Loss'], loc='upper right')
# plt.xlabel('Training Iterations')
# plt.ylabel('Reg Loss')
#
# fig = plt.figure()
# plt.plot(train_counter, train_total_loss_list, color='blue')
# plt.legend(['Training Loss'], loc='upper right')
# plt.xlabel('Training Iterations')
# plt.ylabel('Total Loss')


fig = plt.figure()
plt.plot(epoch_list, epoch_total_loss_list, color='blue')
plt.legend(['Training Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Average Training Loss per Epoch')

# Test Loss
fig = plt.figure()
plt.plot(epoch_list, test_loss_list, color='red')
plt.legend(['Validation Loss'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Total Loss')

# Accuracy
fig = plt.figure()
plt.plot(epoch_list, accuracy_list, color='red')
plt.legend(['Accuracy'], loc='upper left')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()