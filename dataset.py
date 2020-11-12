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

    def __init__(self, root, frames_per_clip, step_between_clips=1,
                 frame_rate=None, train=True, transform=None,
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
        video, audio, _, video_idx = self.video_clips.get_clip(idx)
        sample_index = self.indices[video_idx]
        _, class_index = self.samples[sample_index]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, class_index

from dataset import *
if __name__ == '__main__':
    train_dataset = HMDB51("data", frames_per_clip=5)
    test_dataset = HMDB51("data", frames_per_clip=5, train=False)
    print()