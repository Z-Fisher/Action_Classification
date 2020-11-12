import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
import random
import torchvision


class TwoStream(torch.nn.Module):

    def __init__(self, device='cuda'):
        # Initialize the two-stream layers
        super(TwoStream, self).__init__()

        self.device = device

        # Define Spatial Layer


        # Define Temporal Layers



    def forward(self, X):
        pass


    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
        pass


    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        pass



if __name__ == "__main__":
    pass