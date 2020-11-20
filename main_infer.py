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



# Training
def train(epoch):
    model.train()
    counter = 0
    train_loss = 0
    log_interval = 25

    # Data loop
    epoch_class_loss = []
    epoch_reg_loss = []
    epoch_total_loss = []
    for iter, batch in enumerate(train_loader, 0):

        images, label, mask, bbox, indexes = [batch[i] for i in range(len(batch))]
        images = images.to(device)

        optimizer.zero_grad()

        # Take the features from the backbone
        backout = backbone(images)

        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(images, [(800, 1088)] * images.shape[0])
        # Then we pass the image list and the backbone output through the rpn
        rpnout = rpn(im_lis, backout)

        # The final output is
        # A list of proposal tensors: list:len(bz){(keep_topK,4)}
        proposals = [proposal[0:keep_topK, :] for proposal in rpnout[0]]
        # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
        fpn_feat_list = list(backout.values())

        for i in range(len(proposals)):
            proposals[i] = proposals[i].to(device)

        for i in range(len(fpn_feat_list)):
            fpn_feat_list[i] = fpn_feat_list[i].to(device)

        # create ground truth
        labels, regressor_target = boxhead.create_ground_truth(proposals, label, bbox)

        # ROI align
        feature_vectors = boxhead.MultiScaleRoiAlign(fpn_feat_list, proposals, 7)
        feature_vectors = feature_vectors.to(device)

        # pass feature vectors to box head
        class_logits, box_pred = boxhead(feature_vectors)

        if torch.cuda.is_available():
            class_logits = class_logits.cuda()
            box_pred = box_pred.cuda()
            labels = labels.cuda()
            regressor_target = regressor_target.cuda()

        # calculate losses
        loss, loss_c, loss_r = boxhead.compute_loss(class_logits, box_pred, labels, regressor_target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Logging Interval
        epoch_class_loss.append(loss_c.item())
        epoch_reg_loss.append(loss_r.item())
        epoch_total_loss.append(loss.item())

        if counter % log_interval == log_interval - 1 or counter == 0:
            print('Epoch: ', epoch, ', Batch: ', iter, ', total loss: ', loss.item(), ', class loss: ', loss_c.item(),
                  ', reg loss: ', loss_r.item())
            train_class_loss_list.append(loss_c.item() / (iter + 1) * batch_size)
            train_reg_loss_list.append(loss_r.item() / (iter + 1) * batch_size)
            train_total_loss_list.append(train_loss / (iter + 1) * batch_size)
            train_counter.append((iter + 1) * batch_size + epoch * len(train_loader.dataset))
        counter += 1

    avg_class_loss = sum(epoch_class_loss) / len(epoch_class_loss)
    avg_reg_loss = sum(epoch_reg_loss) / len(epoch_reg_loss)
    avg_total_loss = sum(epoch_total_loss) / len(epoch_total_loss)
    epoch_class_loss_list.append(avg_class_loss)
    epoch_reg_loss_list .append(avg_reg_loss)
    epoch_total_loss_list.append(avg_total_loss)
    print('Epoch: ', epoch, ', avg class loss: ', avg_class_loss, ', avg reg loss: ', avg_reg_loss, ', avg total loss: ', avg_total_loss)

# Evaluating
def test():
    boxhead.eval()
    test_loss = 0

    # Data Loop
    with torch.no_grad():
        for iter, batch in enumerate(test_loader, 0):

            images, label, mask, bbox, indexes = [batch[i] for i in range(len(batch))]
            images = images.to(device)

            # Take the features from the backbone
            backout = backbone(images)

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images, [(800, 1088)] * images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpnout = rpn(im_lis, backout)

            # The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals = [proposal[0:keep_topK, :] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list = list(backout.values())

            for i in range(len(proposals)):
                proposals[i] = proposals[i].to(device)

            for i in range(len(fpn_feat_list)):
                fpn_feat_list[i] = fpn_feat_list[i].to(device)

            # ROI align
            feature_vectors = boxhead.MultiScaleRoiAlign(fpn_feat_list, proposals, 7)
            feature_vectors = feature_vectors.to(device)

            # pass feature vectors to box head
            class_logits, box_pred = boxhead(feature_vectors)

            # create ground truth
            labels, regressor_target = boxhead.create_ground_truth(proposals, label, bbox)

            if torch.cuda.is_available():
                class_logits = class_logits.cuda()
                box_pred = box_pred.cuda()
                labels = labels.cuda()
                regressor_target = regressor_target.cuda()

            # calculate losses
            loss, loss_c, loss_r = boxhead.compute_loss(class_logits, box_pred, labels, regressor_target)

            test_loss += loss.item()

    # Log
    test_loss_list.append(test_loss / len(test_loader.dataset))
    print('Avg Validation Loss: ', test_loss / len(test_loader.dataset))



# Paths
DATA_FOLDER = '/content/drive/MyDrive/680_final/data'
EPOCH_SAVE_PREFIX = '/content/drive/My Drive/680_final/models/'

# Data
train_dataset = HMDB51(DATA_FOLDER, frames_per_clip=5)
test_dataset = HMDB51(DATA_FOLDER, frames_per_clip=5, train=False)

batch_size = 2
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
learning_rate = 0.0007
spatial = SpatialStream()
spatial.to(device)
optimizer = optim.SGD(spatial.parameters(), lr=learning_rate, momentum=)

# initialize weights
for m in spatial.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean = 0, std = 0.01)
        torch.nn.init.constant_(m.bias, 0)

# Epochs
num_epochs = 50

# Logging setup: train
train_loss_list = []
epoch_loss_list = []
train_counter = []

# Logging setup: test
test_loss_list = []
epoch_list = np.arange(num_epochs)

# epoch loop
for epoch in range(num_epochs):

    # Train & Validate
    train(epoch)
    test()

    # Adjust Learning Rate
    if epoch == 8 or epoch == 13:
        pass

    # Save Model Version
    save_path = os.path.join(EPOCH_SAVE_PREFIX, 'spatial_epoch' + str(epoch))
    torch.save({
        'epoch': epoch,
        'train_total_loss_list': train_loss_list,
        'epoch_total_loss_list': epoch_loss_list,
        'test_loss_list': test_loss_list,
        'train_counter': train_counter,
        'model_state_dict': boxhead.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)

    print("Epoch %d/%d Completed" % (epoch, num_epochs - 1))
