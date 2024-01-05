import random
from .utils import cumsum, cumdiff
from torch.utils.data import Dataset


class ClassificationSubset(Dataset):
    """
    Args:
        - dataset (PyTorch Dataset)
        - classes (int, sequence[int] or sequence[str], optional):
            - int n: samples n classes.
            - sequence[int]: selects classes by ids.
            - sequence[str]: selects classes by names.
        - n_samples (int, optional):
            Samples a total of n_samples points while maintaining classes proportions.
        - classes_seed (int, optional): Used only when classes is int.
        - samples_seed (int, optional)
    """

    def __init__(self, dataset, classes=None, n_samples=None, classes_seed=None, samples_seed=None):
        self.dataset = dataset

        self.name = f"Subset({dataset.name})"
        self.channels, self.height, self.width = dataset.channels, dataset.height, dataset.width
        self.mean, self.std = dataset.mean, dataset.std
        self.split = dataset.split

        self.parent_class_ids = self._get_class_ids(classes, classes_seed)
        self.parent_class_ids = list(set(self.parent_class_ids))
        self.parent_class_ids.sort()
        from_parent_class_idx = {c: i for i, c in enumerate(self.parent_class_ids)}

        self.classes = [self.dataset.classes[i] for i in self.parent_class_ids]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.n_classes = len(self.classes)

        self.parent_sample_ids = self._get_sample_ids(n_samples, samples_seed)
        self.targets = [from_parent_class_idx[self.dataset.targets[i]] for i in self.parent_sample_ids]

        self.ids_by_class = [[] for _ in range(self.n_classes)]
        for i, target in enumerate(self.targets):
            self.ids_by_class[target].append(i)
        self.n_samples_by_class = [len(ids) for ids in self.ids_by_class]
        self.avg_n_samples_by_class = len(self) / self.n_classes
    
    def _get_class_ids(self, classes, seed=None):
        tot_classes = self.dataset.n_classes
        if classes is None:
            return list(range(tot_classes))
        elif isinstance(classes, int):
            if seed is not None:
                random.seed(seed)
            return random.sample(range(tot_classes), classes)
        elif isinstance(classes[0], int):
            assert max(classes) < tot_classes
            return classes
        elif isinstance(classes[0], str):
            return [self.dataset.class_to_idx[c] for c in classes]
        else:
            raise ValueError("classes must be None, int, sequence[int] or sequence[str]")
    
    def _get_sample_ids(self, n_samples, seed=None):
        if seed is not None:
            random.seed(seed)
        tot_samples_by_class = [self.dataset.n_samples_by_class[c] for c in self.parent_class_ids]
        tot_samples = sum(tot_samples_by_class)
        if n_samples is None:
            n_samples = tot_samples
        assert n_samples <= tot_samples
        cum_tot_samples_by_class = cumsum(tot_samples_by_class)
        cum_n_samples_by_class = [(n_samples * k) // tot_samples for k in cum_tot_samples_by_class]
        n_samples_by_class = cumdiff(cum_n_samples_by_class)
        ids_by_class = [self.dataset.ids_by_class[c] for c in self.parent_class_ids]
        res = [random.sample(ids, n_samples_by_class[c]) for c, ids in enumerate(ids_by_class)]
        return sum(res, [])
    
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