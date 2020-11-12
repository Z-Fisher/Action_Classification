import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        pass

    def __getitem__(self, index):
        pass

    def pre_process_batch(self, img, mask, bbox):
        pass

    def __len__(self):
        pass



if __name__ == '__main__':
    # file path and make a list
    path_dir = 'zach'

    if path_dir == 'akhil':
        imgs_path = 'C:/Users/Akhil/OneDrive - PennO365/Documents/UPenn/Fall 2020/CIS_680/HW4/data/hw3_mycocodata_img_comp_zlib.h5'
        masks_path = 'C:/Users/Akhil/OneDrive - PennO365/Documents/UPenn/Fall 2020/CIS_680/HW4/data/hw3_mycocodata_mask_comp_zlib.h5'
        labels_path = 'C:/Users/Akhil/OneDrive - PennO365/Documents/UPenn/Fall 2020/CIS_680/HW4/data/hw3_mycocodata_labels_comp_zlib.npy'
        bboxes_path = 'C:/Users/Akhil/OneDrive - PennO365/Documents/UPenn/Fall 2020/CIS_680/HW4/data/hw3_mycocodata_bboxes_comp_zlib.npy'
    elif path_dir == 'zach':
        imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
        masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
        labels_path = '../data/hw3_mycocodata_labels_comp_zlib.npy'
        bboxes_path = '../data/hw3_mycocodata_bboxes_comp_zlib.npy'
    elif path_dir == 'colab':
        imgs_path = '/content/drive/My Drive/SOLO/data/hw3_mycocodata_img_comp_zlib.h5'
        masks_path = '/content/drive/My Drive/SOLO/data/hw3_mycocodata_mask_comp_zlib.h5'
        labels_path = '/content/drive/My Drive/SOLO/data/hw3_mycocodata_labels_comp_zlib.npy'
        bboxes_path = '/content/drive/My Drive/SOLO/data/hw3_mycocodata_bboxes_comp_zlib.npy'

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)






