from torch.utils.data.sampler import Sampler, RandomSampler
import itertools
import numpy as np
from torch.utils.data.sampler import BatchSampler
import random

import pdb

class SequentialNRepeatSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, N_repeats = 1):
        self.data_source = data_source
        self.N_repeats =  N_repeats

    def __iter__(self):
        return iter(itertools.chain.from_iterable(itertools.repeat(x, self.N_repeats) for x in range(len(self.data_source))))

    def __len__(self):
        return len(self.data_source)*self.N_repeats

class RandomSequentialNRepeatSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, N_repeats = 1):
        self.data_source = data_source
        self.N_repeats =  N_repeats

    def __iter__(self):
        iterator = list(itertools.chain.from_iterable(itertools.repeat(x, self.N_repeats) for x in range(len(self.data_source))))
        random.shuffle(iterator)
        return iter(iterator)

    def __len__(self):
        return len(self.data_source)*self.N_repeats
        
class RandomSampler_own(RandomSampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """
    def __len__(self):
        return self.num_samples

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, train):
        self.labels = np.array([s[1] for s in dataset.samples])
        self.train = train
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
        
class BalancedBatchSamplerReplace(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples, N_repeats = 100):
        self.labels = np.array([s["_idx"] for s in dataset.samples])
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        self.N_repeats  = N_repeats

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset)*self.N_repeats:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend([self.label_to_indices[class_]]*self.n_samples)
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset)*self.N_repeats//self.batch_size