import glob
import os

from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

class HMDB51(VisionDataset):
    """
    Internally, it uses a VideoClips object to handle clip creation.
    Args:
        root (string): Root directory of the HMDB51 Dataset.
        frames_per_clip (int): Number of frames in a clip.
        step_between_clips (int): Number of frames between each clip.
        train (bool, optional): If ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that takes in a TxHxWxC video
            and returns a transformed version.
    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """
    def __init__(self, root=""):
        super(HMDB51, self).__init__(root)

    def init_data(self, root, frames_per_clip, step_between_clips=6,
                 frame_rate=6, train=True, transform=None,
                 _precomputed_metadata=None, num_workers=1, _video_width=0,
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(HMDB51, self).__init__(root)
        extensions = ('avi',)
        if train:
            root = root + "/train"
        else:
            root = root + "/test"
        classes = sorted(list_dir(root))
        class_to_idx = {class_: i for (i, class_) in enumerate(classes)}
        print(class_to_idx)
        self.samples = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            for root_curr, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root_curr, fname)
                    if os.path.isfile(path):
                        item = path, class_index
                        self.samples.append(item)

        video_paths = [path for (path, _) in self.samples]
        video_clips = VideoClips(
            video_paths,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        self.train = train
        self.classes = classes
        self.video_clips_metadata = video_clips.metadata
        self.indices = self.get_indices(video_paths)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips_metadata

    def get_indices(self, video_list):
        indices = []
        for video_index, video_path in enumerate(video_list):
            indices.append(video_index)
        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = video.permute(0, 3, 1, 2)
            video = self.transform(video)
            video = video.permute(0, 2, 3, 1)

        return video, class_index, sample_index

    # Get the state dict of the dataset
    def state_dict(self):
        state = {"video_clips": self.video_clips,
                 "indices": self.indices,
                 "samples": self.samples,
                 "transform": self.transform,
                 "metadata": self.video_clips_metadata}
        return state

    # Load from state dict file for the dataset
    def load_state_dict(self, state):
        self.video_clips = state["video_clips"]
        self.indices = state["indices"]
        self.samples = state["samples"]
        self.transform = state["transform"]
        self.video_clips_metadata = state["metadata"]

import torch
from torch.utils.data import DataLoader


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.w_frame = 224
        self.h_frame = 224

    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        video_list = []
        label_list = []
        index_list = []

        for video, cl_index, s_index in batch:
            video_list.append(video)
            label_list.append(cl_index)
            index_list.append(s_index)

        data = {"videos": torch.stack(video_list),
                "labels": label_list,
                "indexes": index_list
                }

        return data

    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)

import cv2 as cv
import numpy as np

# Calculate optical flow and trajectory stack over frames
def getOpticalFlow(videos):
    bz = videos.size(0)

    flow_list_batch = []

    for b in range(bz):
        images = videos[b]

        w = images[0].size(0)
        h = images[0].size(1)
        num_fr = images.size(0)

        prev = cv.cvtColor(images[0].numpy(), cv.COLOR_BGR2GRAY)
        flow_list = []

        # Calculate flows
        for i in range(1, num_frames):
            next = cv.cvtColor(images[i].numpy(), cv.COLOR_BGR2GRAY)
            flow_list.append(cv.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0))
            prev = next.copy()

        # Perform mean flow subtraction between subsequent frames
        for i in range(len(flow_list)):
            curr_flow = flow_list[i]
            mean = np.mean(curr_flow, axis=2)
            flow_list[i][:,:,0] -= mean
            flow_list[i][:,:,1] -= mean

        # Visualize flows
        for flow in flow_list:
            hsv = np.zeros_like(images[0])
            hsv[..., 1] = 255

            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow("Flow", bgr)
            cv.waitKey(0)

        flow_list_batch.append(np.stack(flow_list, axis=-1).reshape(w, h, 2 * (num_fr - 1)))

    flows = np.stack(flow_list_batch, axis=0)
    return torch.Tensor(flows)


import torchvision.transforms as tf
import matplotlib.pyplot as plt
import matplotlib
import pickle
import h5py
matplotlib.use('TkAgg')

if __name__ == '__main__':
    DATA_FOLDER = "data"
    num_frames = 1

    # Save the training data in files for each batch
    transform = tf.Compose([tf.Resize(256), tf.RandomCrop(224)])

    # Load dataset
    train_dataset = HMDB51(DATA_FOLDER)
    train_dataset.init_data(DATA_FOLDER, frames_per_clip=num_frames, transform=transform)

    print(len(train_dataset))

    path_train = DATA_FOLDER + "/dataset_{}/train/".format(num_frames)
    batch_size_tr = 128

    path_test = DATA_FOLDER + "/dataset_{}/test/".format(num_frames)
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size_tr, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()

    n = len(train_loader)

    ## Create h5py training dataset (too slow)
    # f = h5py.File(DATA_FOLDER + '/dataset_{}/train_{}.h5'.format(num_frames, num_frames), 'w')
    # dset_img = f.create_dataset("images", (len(train_dataset), 224, 224, 3), compression="gzip", compression_opts=6)
    # dset_label = f.create_dataset("labels", (len(train_dataset), ), compression="gzip", compression_opts=6)

    state_dict = {"videos": [], "labels": [], "indexes": []}
    for iter, data in enumerate(train_loader, 0):
        state_dict['videos'] = data["videos"]
        state_dict['labels'] = data["labels"]

        # Save batch in own file
        torch.save(state_dict, path_train + "train_b{}.pt".format(iter), pickle_protocol=pickle.HIGHEST_PROTOCOL)
        state_dict = {"videos": [], "labels": [], "indexes": []}

        ## Update h5py dataset
        # dset_img[iter*batch_size_tr:(iter*batch_size_tr)+l] = videos[:,0]
        # dset_label[iter*batch_size_tr:(iter*batch_size_tr)+l] = labels[:]

        print("Saved batch {}/{}".format(iter + 1, n))

    # f.close()

    # Save the test data in files for each batch
    transform = tf.Compose([tf.Resize(256), tf.CenterCrop(224)])
    batch_size_test = 128

    test_dataset = HMDB51(DATA_FOLDER)
    test_dataset.init_data(DATA_FOLDER, frames_per_clip=num_frames, train=False, transform=transform)

    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    n = len(test_loader)

    ## Create h5py testing dataset (too slow)
    # f = h5py.File(DATA_FOLDER + '/dataset_{}/test_{}.h5'.format(num_frames, num_frames), 'w')
    # dset_img = f.create_dataset("images", (len(test_dataset), 224, 224, 3), dtype=np.uint8, compression="gzip", compression_opts=6)
    # dset_label = f.create_dataset("labels", (len(test_dataset),), dtype=np.uint8, compression="gzip", compression_opts=6)

    state_dict = {"videos": [], "labels": [], "indexes": []}
    for iter, data in enumerate(test_loader, 0):
        state_dict['videos'] = data["videos"]
        state_dict['labels'] = data["labels"]

        # Save batch in own file
        torch.save(state_dict, path_test + "test_b{}.pt".format(iter), pickle_protocol=pickle.HIGHEST_PROTOCOL)
        state_dict = {"videos": [], "labels": [], "indexes": []}

        ## Update h5py dataset
        # dset_img[iter*batch_size_test:(iter*batch_size_test)+l] = videos[:,0].numpy()
        # dset_label[iter*batch_size_test:(iter*batch_size_test)+l] = labels[:]
        # f.flush()

        print("Saved batch {}/{}".format(iter + 1, n))

    # f.close()

    ## Save HMDB51 state dict to quickly load video classes/paths

    #torch.save(train_dataset.state_dict(), DATA_FOLDER + "/train_dataset_{}.pt".format(num_frames))
    #torch.save(test_dataset.state_dict(), DATA_FOLDER + "/test_dataset_{}.pt".format(num_frames))

    #train_state = torch.load(DATA_FOLDER + "/train_dataset_10.pt")
    #test_state = torch.load(DATA_FOLDER + "/test_dataset_10.pt")
