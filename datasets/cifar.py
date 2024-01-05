import os
import pickle
from PIL import Image
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    """
    Args:
        - root (str)
        - train (bool)
        - transform (callable, optional)
    """

    name = "CIFAR-10"

    base_folder = "cifar-10-batches-py"

    train_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_list = [
        "test_batch",
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
    }

    channels, height, width = 3, 32, 32 # original size
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    def __init__(self, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        self.split = 'train' if train else 'validation'

        files_list = self.train_list if self.train else self.test_list
        self._load_data(files_list)
        self._load_meta()

        self.ids_by_class = [[] for _ in range(self.n_classes)]
        for i, target in enumerate(self.targets):
            self.ids_by_class[target].append(i)
        self.n_samples_by_class = [len(ids) for ids in self.ids_by_class]
        self.avg_n_samples_by_class = len(self) / self.n_classes
    
    def _load_data(self, files_list):
        self.data = []
        self.targets = []
        for filename in files_list:
            filepath = os.path.join(self.root, self.base_folder, filename)
            with open(filepath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                key = "labels" if "labels" in entry else "fine_labels"
                self.targets.extend(entry[key])
        self.data = np.vstack(self.data).reshape(-1, self.channels, self.height, self.width)
        self.data = np.moveaxis(self.data, 1, -1) # channel last

    def _load_meta(self):
        filepath = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(filepath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

    def __len__(self):
        return len(self.data)


class CIFAR100(CIFAR10):

    name = "CIFAR-100"

    base_folder = "cifar-100-python"

    train_list = [
        "train",
    ]
    test_list = [
        "test",
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
    }