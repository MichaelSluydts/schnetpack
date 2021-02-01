import torch,tqdm
from torch._six import int_classes as _int_classes
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import multiprocessing.dummy as mp
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        if int(num_samples) == 0:
            num_samples = None
        self.num_samples = num_samples
        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")
        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:

            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

class StratifiedSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samplies to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py
    """

    def __init__(self, data_source, replacement=True, num_samples=3200, bins = None, nbins = 8):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.nbins = nbins

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))
        if bins == None:
            if os.path.isfile(self.data_source.asedb_path + '.bins.npy'):
                self.bins = np.load(self.data_source.asedb_path + '.bins.npy')
            else:
                self.bins = np.zeros(len(self.data_source))
                self.energy = np.zeros(len(self.data_source))
                self.get_bins()
        else:
            self.bins = bins

    def __iter__(self):
        n = len(self.data_source)
        nsplits=int(n/self.num_samples)
        print('nsplits',nsplits)
        s = StratifiedShuffleSplit(n_splits=int(self.bins.size/self.nbins),test_size=1/nsplits)
        X = torch.randn(nsplits*self.num_samples,2).numpy()
        y = self.bins[0:nsplits*self.num_samples]
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return iter(test_index)

    def __len__(self):
        return self.num_samples

    def gather_energy(self,i):
        total = len(self.data_source)
        piece = int(np.floor(total/9.))
        if i == 0:
            start = 0
        else:
            start = i*piece

        if i == 8:
            end = total
        else:
            end = start + piece

        r = end - start
        temp = np.zeros(int(r))
        for j in tqdm.tqdm(range(r)):
            temp[j] = self.data_source.get_atoms(int(j)).get_potential_energy()
        self.energy[start:end] = temp

    def get_bins(self):
        p=mp.Pool(9)
        tqdm.tqdm(p.map(self.gather_energy,range(9)))
        p.close()
        p.join()
        df = pd.DataFrame({'energy' : self.energy})
        self.nbins = self.nbins if self.nbins < 16 else 16
        df['Ebin'] = pd.qcut(df['energy'].values,q=self.nbins, duplicates='drop', labels=False)
        self.bins = df['Ebin'].values
        np.save(self.data_source.asedb_path + '.bins.npy',self.bins)
        return True


class StratifiedSamplerOld(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samplies to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py
    """

    def __init__(self, data_source, replacement=True, num_samples=3200, bins = None, nbins = 8):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples
        self.nbins = nbins

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if bins == None:
            if os.path.isfile(self.data_source.asedb_path + '.bins.npy'):
                self.bins = np.load(self.data_source.asedb_path + '.bins.npy')
            else:
                self.bins = np.zeros(len(self.data_source))
                self.energy = np.zeros(len(self.data_source))
                self.get_bins()
        else:
            self.bins = bins

    def __iter__(self):
        n = len(self.data_source)
        nsplits=int(n/self.num_samples)
        s = StratifiedShuffleSplit(n_splits=int(self.bins.size/self.nbins),test_size=1/nsplits)
        X = torch.randn(nsplits*self.num_samples,2).numpy()
        y = self.bins[0:nsplits*self.num_samples]
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return iter(test_index)

    def __len__(self):
        return self.num_samples

    def gather_energy(self,i):
        total = len(self.data_source)
        piece = int(np.floor(total/9.))
        if i == 0:
            start = 0
        else:
            start = i*piece

        if i == 8:
            end = total
        else:
            end = start + piece

        r = end - start
        temp = np.zeros(int(r))
        for j in tqdm.tqdm(range(r)):
            temp[j] = self.data_source.get_atoms(int(j)).get_potential_energy()
        self.energy[start:end] = temp

    def get_bins(self):
        p=mp.Pool(9)
        tqdm.tqdm(p.map(self.gather_energy,range(9)))
        p.close()
        p.join()
        df = pd.DataFrame({'energy' : self.energy})
        df['Ebin'] = pd.qcut(df['energy'].values,q=self.nbins,labels=range(self.nbins))
        self.bins = df['Ebin'].values
        np.save(self.data_source.asedb_path + '.bins',self.bins)
        return True
