import os
import random
import utils
from torch.utils.data import Dataset
from .imagenet import ImageNet


class MiniImageNetPlus(Dataset):
    """
    Args:
        - root (str)
        - train (bool)
        - transform (callable, optional)
    """

    name = "mini-ImageNet+"
    
    classes_file = "class_index.json"

    seed = 0
    train_samples_by_class = 1200
    val_samples_by_class = 100

    def __init__(self, root, train, transform=None, exclude_ids=None):
        self.dataset = ImageNet(root=root['imagenet'], train=True)
        self.channels, self.height, self.width = self.dataset.channels, self.dataset.height, self.dataset.width
        self.mean, self.std = self.dataset.mean, self.dataset.std

        self.root = root['mini_imagenet']
        self.train = train
        self.transform = transform

        self.split = 'train' if train else 'validation'

        self.n_samples_by_class = self.train_samples_by_class if train else self.val_samples_by_class
        self.exclude_ids = set(exclude_ids) if exclude_ids else None
        if not train: assert self.exclude_ids is not None

        self._load_meta()
        self._sample_ids()

        self.ids_by_class = [[] for _ in range(self.n_classes)]
        for i, target in enumerate(self.targets):
            self.ids_by_class[target].append(i)
        self.n_samples_by_class = [len(ids) for ids in self.ids_by_class]
        self.avg_n_samples_by_class = len(self) / self.n_classes
    
    def _load_meta(self):
        js = utils.load_json(os.path.join(self.root, self.classes_file))
        self.parent_class_ids = [self.dataset.syn_to_class_idx[syn] for syn in js]
        self.from_parent_class_idx = {c: i for i, c in enumerate(self.parent_class_ids)}
        self.classes = [self.dataset.syn_to_name[syn] for syn in js]
        self.syn_to_class_idx = {s: i for i, s in enumerate(js)}
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)
    
    def _sample_ids(self):
        random.seed(self.seed)
        parent_ids_by_class = [self.dataset.ids_by_class[c] for c in self.parent_class_ids]
        self.parent_sample_ids = []
        for ids in parent_ids_by_class:
            if self.train:
                n_samples = min(self.train_samples_by_class, len(ids) - self.val_samples_by_class)
            else:
                n_samples = self.val_samples_by_class
            if self.exclude_ids: ids = list(set(ids) - self.exclude_ids)
            self.parent_sample_ids.extend(random.sample(ids, n_samples))
        self.targets = [self.from_parent_class_idx[self.dataset.targets[i]] for i in self.parent_sample_ids]
    
    @property
    def sample_ids(self):
        return self.parent_sample_ids
    
    @property
    def transform(self):
        return self.dataset.transform
    
    @transform.setter
    def transform(self, value):
        self.dataset.transform = value
    
    def __getitem__(self, idx):
        sample, _ = self.dataset[self.parent_sample_ids[idx]]
        target = self.targets[idx]
        return sample, target
    
    def __len__(self):
        return len(self.parent_sample_ids)