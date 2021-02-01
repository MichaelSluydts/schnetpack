import logging
import os

import numpy as np
import torch
from ase.db import connect
from schnetpack2.environment import SimpleEnvironmentProvider, collect_atom_triples
from schnetpack2.custom.environment import collect_atom_triples_offsets
from schnetpack2.data import AtomsData, Structure
from torch.utils.data import Dataset, DataLoader
import pdb

cuda = torch.cuda.is_available()

PROP_MAP = {'1JHC':0, '1JHN':1, '2JHC':2, '2JHH':3, '2JHN':4, '3JHC':5, '3JHH':6, '3JHN':7}

class Champs(AtomsData):
    def __getitem__(self, idx):
        # get row
        if self.subset is None:
            row = self.asedb.get(int(idx) + 1)
        else:
            row = self.asedb.get(int(self.subset[idx]) + 1)
        at = row.toatoms()

        # extract properties
        properties = {}
        for p in self.properties:
            # Capture exception for ISO17 where energies are stored directly in the row
            if p in row.data:
                prop = row.data[p]
            else:
                prop = row[p]
                
            if p=="type":
                prop = [PROP_MAP[prp] for prp in prop]
 
            try:
                prop.shape
            except AttributeError as e:
                prop = np.array([prop], dtype=np.float32)
            properties[p] = torch.FloatTensor(prop)

        # extract/calculate structure
        properties[Structure.Z] = torch.LongTensor(at.numbers.astype(np.int))

        positions = at.positions.astype(np.float32)
        if self.centered:
            positions -= at.get_center_of_mass()
        properties[Structure.R] = torch.FloatTensor(positions)

        properties[Structure.cell] = torch.FloatTensor(at.cell.astype(np.float32))

        # get atom environment
        nbh_idx, offsets = self.environment_provider.get_environment(idx, at)

        properties[Structure.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
        properties[Structure.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
        properties['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

        if self.collect_triples:
            nbh_idx_j, nbh_idx_k = collect_atom_triples(nbh_idx)
            properties[Structure.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            properties[Structure.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))

        return properties
