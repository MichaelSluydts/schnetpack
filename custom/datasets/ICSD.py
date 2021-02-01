import os

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import eV

from schnetpack2.data import AtomsData
from schnetpack2.environment import ASEEnvironmentProvider

__all__ = [
    'ICSD'
]


class ICSD(AtomsData):
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
        self.collect_triples = collect_triples
        self.dbpath = os.path.join(self.path, 'icsd.db')

        environment_provider = ASEEnvironmentProvider(cutoff)

        super(ICSD, self).__init__(self.dbpath, subset, properties, environment_provider, False)

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]

        return ICSD(self.path, self.cutoff, subset=subidx, properties=self.properties,
                                collect_triples=self.collect_triples)