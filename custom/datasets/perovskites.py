import logging
import os

import numpy as np
import torch
from ase.db import connect
from schnetpack2.environment import ASEEnvironmentProvider, collect_atom_triples
from schnetpack2.custom.environment import collect_atom_triples_offsets
from schnetpack2.custom.data import AtomsData, Structure
from torch.utils.data import Dataset, DataLoader
import pdb

cuda = torch.cuda.is_available()

class Perovskites(AtomsData):
    """ Materials Project data repository of bulk crystals.

        This class adds convenience functions to download Materials Project data into pytorch.

        Args:
            path (str): path to directory containing mp database.
            cutoff (float): cutoff for bulk interactions
            apikey (str): materials project key needed to download the data (default: None)
            download (bool): enable downloading if database does not exists (default: True)
            subset (list): indices of subset. Set to None for entire dataset (default: None)
            properties (list): properties, e.g. formation_energy_per_atom

    """

    def __init__(self, path, cutoff, properties = [], subset=None, collect_triples = False):
        self.path = path
        self.cutoff = cutoff
        self.properties = properties
        self.collect_triples = collect_atom_triples_offsets
        if self.path.rsplit(".",1)[-1] == "db": self.dbpath = self.path
        else: self.dbpath = os.path.join(self.path, 'perovskites.db')

        environment_provider = ASEEnvironmentProvider(cutoff)

        super(Perovskites, self).__init__(self.dbpath, subset, properties, environment_provider, collect_triples)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return Perovskites(self.path, self.cutoff, subset=subidx, properties=self.properties,
                                collect_triples=self.collect_triples)
