import random
import pandas as pd
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Dataset


import pdb
        
class Subset_class(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, n_classes):
        self.dataset = dataset
        self.indices = indices
        self.c       = n_classes
        self.samples = [dataset.samples[i] for i in indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
        
#def _get_triplets_init(dataset):
#    labels = np.array([dataset.asedb.get_atoms(i+1).get_total_energy()//0.25 for i in range(len(dataset.asedb))]).astype(int)
#    labels_set = set(labels)
#    label_to_indices = {label: np.where(labels == label)[0]
#                             for label in labels_set}
#
#    random_state = np.random.RandomState(29)
#
#    triplets = [[i,
#                 random_state.choice(label_to_indices[labels[i]]),
#                 random_state.choice(label_to_indices[
#                                         np.random.choice(
#                                             list(labels_set - set([labels[i]]))
#                                         )
#                                     ])
#                 ]
#                for i in range(len(dataset))]
#                
#    return triplets

def _get_triplets_init(dataset, n_permutations):
    labels = np.hstack([[i]*n_permutations for i in range(len(dataset))])#np.array([dataset.asedb.get_atoms(i+1).get_total_energy()//0.25 for i in range(len(dataset.asedb))]).astype(int)
    labels_set = set(labels)
    label_to_indices = {label: np.where(labels == label)[0]
                             for label in labels_set}

    random_state = np.random.RandomState(29)

    triplets = [[i,
                 random_state.choice(label_to_indices[labels[i]]),
                 random_state.choice(label_to_indices[
                                         np.random.choice(
                                             list(labels_set - set([labels[i]]))
                                         )
                                     ])
                 ]
                for i in range(len(dataset)*n_permutations)]
                
    return triplets

class TripletsOfflineDataset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, n_permutations, n_classes=None, transform=None):
        self.dataset        = dataset
        self.n_permutations = n_permutations
        self.triplets       = _get_triplets_init(dataset, n_permutations)
        self.c              = n_classes#dataset.c
        self.transform      = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            return [self.transform(self.dataset[triplet//self.n_permutations], idx) for triplet in self.triplets[idx]]
        else:
            return [self.dataset[triplet//self.n_permutations] for triplet in self.triplets[idx]]

    def __len__(self):
        return len(self.triplets)
        
    def _set_triplets(self, triplets):
        self.triplets = triplets
        
class TripletsOnlineDataset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, transform=None):
        self.samples        = dataset
        self.c              = len(dataset)
        self.transform      = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.samples[idx])
        else:
            return self.samples[idx]

    def __len__(self):
        return len(self.samples)
        
class CNNOnlineDataset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, N_repeats, transform=None):
        self.samples        = dataset
        self.c              = len(dataset)
        self.transform      = transform

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.samples[idx])
        else:
            return self.samples[idx]

    def __len__(self):
        return len(self.samples)