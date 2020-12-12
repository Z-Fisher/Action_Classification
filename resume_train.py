import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import *
from SpatialStream import *
import torchvision.transforms as tf



# Training
def train(epoch):
    spatial.train()
    counter = 0
    train_loss = 0
    log_interval = 25

    epoch_loss = []
    log_int_loss = 0
    for iter, data in enumerate(train_loader, 0):

        videos = data["videos"]
        labels = torch.tensor(data["labels"])
        indexes = data["indexes"]

        videos = videos.type(torch.FloatTensor)
        videos = videos.to(device)
        # labels = labels.to(device)

        optimizer.zero_grad()

        output = spatial(videos)
        output = output.to(device)
        labels = labels.to(device)

        # calculate losses
        loss = spatial.compute_loss(output, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Logging Interval
        log_int_loss += loss.item()
        epoch_loss.append(loss.item())

        if counter == 0:
            print('Epoch: ', epoch, ', Batch: ', iter, ', loss avg over log interval: ', log_int_loss)
            train_loss_list.append(train_loss / (iter + 1) * batch_size)
            train_counter.append((iter + 1) * batch_size + epoch * len(train_loader.dataset))
            log_int_loss = 0
        elif counter % log_interval == log_interval - 1:
            print('Epoch: ', epoch, ', Batch: ', iter, ', loss avg over log interval: ', log_int_loss / log_interval)
            train_loss_list.append(train_loss / (iter + 1) * batch_size)
            train_counter.append((iter + 1) * batch_size + epoch * len(train_loader.dataset))
            log_int_loss = 0
        counter += 1

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    epoch_loss_list.append(avg_loss)
    print('Epoch: ', epoch, ', avg total loss: ', avg_loss)

# Evaluating
def test():
    spatial.eval()
    test_loss = 0
    correct = 0

    # Data Loop
    with torch.no_grad():
        for iter, data in enumerate(test_loader, 0):
            videos = data["videos"]
            labels = torch.tensor(data["labels"])
            indexes = data["indexes"]

            videos = videos.type(torch.FloatTensor)
            videos = videos.to(device)
            labels = labels.to(device)

            output = spatial(videos)
            output = output.to(device)

            # calculate losses
            loss = spatial.compute_loss(output, labels)

            test_loss += loss.item()

            # calculate number of correct predictions in batch
            correct += sum(torch.argmax(output,1) == labels).item()

            if iter % 100 == 0:
                print ("iter  ", iter)
                print("accuracy so far = ", correct / ((iter + 1) * len(labels)))

    # Log
    test_loss_list.append(test_loss / len(test_loader.dataset))
    accuracy = correct / len(test_loader.dataset)
    print('Avg Validation Loss: ', test_loss / len(test_loader.dataset))
    print('Accuracy: ', accuracy)



# Paths
paths = 'akhil'

if paths == 'akhil':
    DATA_FOLDER = 'data'
elif paths == 'colab':
    DATA_FOLDER = '/content/drive/MyDrive/680_final/data'

EPOCH_SAVE_PREFIX = '/content/drive/My Drive/Shared drives/CIS680 Final Project/models/spatial/'

# Data

# transform = tf.Compose([tf.Resize(256), tf.RandomCrop(224)])
#
# train_dataset = HMDB51(DATA_FOLDER)
# test_dataset = HMDB51(DATA_FOLDER)
#
# train_dataset.init_data(DATA_FOLDER, frames_per_clip=1, transform=transform)
# torch.save(train_dataset.state_dict(), DATA_FOLDER + "/train_dataset_10.pt")
#
# test_dataset.init_data(DATA_FOLDER, frames_per_clip=1, train=False, transform=transform)
# torch.save(test_dataset.state_dict(), DATA_FOLDER + "/test_dataset_10.pt")


train_dataset = HMDB51(DATA_FOLDER)
test_dataset = HMDB51(DATA_FOLDER)

train_dataset.load_state_dict(torch.load(DATA_FOLDER + "/train_dataset_10.pt"))
test_dataset.load_state_dict(torch.load(DATA_FOLDER + "/test_dataset_10.pt"))

batch_size = 2
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
learning_rate = 0.001
spatial = SpatialStream()
spatial.to(device)
optimizer = optim.SGD(spatial.parameters(), lr=learning_rate, momentum=0.9)

# initialize weights
# for m in spatial.modules():
#     if isinstance(m, torch.nn.Linear):
#         torch.nn.init.normal_(m.weight, mean = 0, std = 0.01)
#         torch.nn.init.constant_(m.bias, 0)

# Epochs
num_epochs = 50
epoch_list = np.arange(num_epochs)


# LOAD NETWORK
network_path = 'models/spatial_epoch0'
checkpoint = torch.load(network_path, map_location=torch.device('cpu'))
spatial.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
last_epoch = checkpoint['epoch']

# Logging setup: train
train_loss_list = checkpoint['train_total_loss_list']
epoch_loss_list = checkpoint['epoch_total_loss_list']
test_loss_list = checkpoint['test_loss_list']
train_counter = checkpoint['train_counter']


# epoch loop
for epoch in range(last_epoch + 1, num_epochs):

    # Train & Validate
    # train(epoch)
    test()
    break

    # Adjust Learning Rate
    # if epoch == 8 or epoch == 13:
    #     pass

    # Save Model Version
    save_path = os.path.join(EPOCH_SAVE_PREFIX, 'spatial_epoch' + str(epoch))
    torch.save({
        'epoch': epoch,
        'train_total_loss_list': train_loss_list,
        'epoch_total_loss_list': epoch_loss_list,
        'test_loss_list': test_loss_list,
        'train_counter': train_counter,
        'model_state_dict': spatial.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

    print("Epoch %d/%d Completed" % (epoch, num_epochs - 1))
