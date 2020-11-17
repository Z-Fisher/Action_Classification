import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
import random
import torchvision
import torchvision.models as models
import torch.optim as optim



class SpatialStream(torch.nn.Module):

    def __init__(self, 
                 device='cuda',
                 num_classes=51,
                 dropout_probability=0.5
                 train_resnet=True):

        # Initialize the stream layers
        super(SpatialStream, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Spatial Backbone
        self.spatial = models.resnet50(pretrained=True)
        for param in self.spatial.parameters():
            param.requires_grad = train_resnet  # False: Freezes the weights of the pre-trained model

        # Add to Spatial Backbone
        self.spatial.fc = nn.Sequential(nn.Linear(2048, 1024),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_probability),
                                nn.Linear(1024, self.num_classes),
                                nn.ReLU(),
                                nn.Dropout(p=dropout_probability),
                                nn.Softmax())

    def forward(self, X):
        return self.spatial(X)

    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
        pass

    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        pass

if __name__ == "__main__":

    # Spatial
    spatial_stream = SpatialStream()
    spatial_optimizer = optim.SGD(spatial_stream.parameters(), lr=0.1, momentum=0.9)

    # Temporal
    

