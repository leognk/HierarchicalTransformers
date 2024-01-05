import os
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import utils
from torch.utils.data import Dataset


class ImageNet(Dataset):
    """
    Args:
        - root (str)
        - train (bool)
        - transform (callable, optional)
    """

    name = "ImageNet"

    classes_file = "imagenet_class_index.json"
    val_labels_file = "ILSVRC2012_val_labels.json"

    data_base_folder = "ILSVRC/Data/CLS-LOC"
    train_data_folder = "train"
    val_data_folder = "val"

    channels, height, width = 3, None, None # original size
    mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    def __init__(self, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        self.split = 'train' if train else 'validation'

        self._load_meta()
        self._load_data()

        self.ids_by_class = [[] for _ in range(self.n_classes)]
        for i, target in enumerate(self.targets):
            self.ids_by_class[target].append(i)
        self.n_samples_by_class = [len(ids) for ids in self.ids_by_class]
        self.avg_n_samples_by_class = len(self) / self.n_classes
    
    def _load_meta(self):
        js = utils.load_json(os.path.join(self.root, self.classes_file))
        self.syn_to_name = dict(js.values())
        self.syn_to_class_idx = {}
        self.classes = [None] * len(js)
        for idx, (syn, name) in js.items():
            self.syn_to_class_idx[syn] = int(idx)
            self.classes[int(idx)] = name
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)
    
    def _load_data(self):
        data_folder = self.train_data_folder if self.train else self.val_data_folder
        data_path = os.path.join(self.root, self.data_base_folder, data_folder)
        self.data = []
        self.targets = []
        if self.train:
            for syn in os.listdir(data_path):
                target = self.syn_to_class_idx[syn]
                syn_path = os.path.join(data_path, syn)
                for filename in os.listdir(syn_path):
                    filepath = os.path.join(syn_path, filename)
                    self.data.append(filepath)
                    self.targets.append(target)
        else:
            val_labels = utils.load_json(os.path.join(self.root, self.val_labels_file))
            for filename in os.listdir(data_path):
                filepath = os.path.join(data_path, filename)
                syn = val_labels[filename]
                target = self.syn_to_class_idx[syn]
                self.data.append(filepath)
                self.targets.append(target)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]

    def __len__(self):
        return len(self.data)