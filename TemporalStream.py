import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
import random
import torchvision
import torchvision.models as models
import torch.optim as optim

class TemporalStream(torch.nn.Module):

    def __init__(self,
                 device='cuda',
                 num_classes=51,
                 dropout_probability=0.5):

        # Initialize the stream layers
        super(TemporalStream, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Temporal Stream Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2),
            nn.Relu(),
            nn.MaxPool2d(3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=2),
            nn.Relu(),
            nn.MaxPool2d(3, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.Relu(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.Relu(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.Relu(),
            nn.MaxPool2d(3, stride=2)
        )

        self.fc1 =nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Dropout(p=dropout_probability)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.Dropout(p=dropout_probability)
        )

        self.softmax = nn.Softmax()

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.fc1(X) # TODO check this, probably needs X to be resized
        X = self.fc2(X) # TODO another linear layer needed for output?
        X = self.softmax(X)
        return X

    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
        pass

    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        pass


if __name__ == "__main__":
    # Temporal
    temporal_stream = TemporalStream()
    temporal_optimizer = optim.SGD(temporal_stream.parameters(), lr=0.1, momentum=0.9)




