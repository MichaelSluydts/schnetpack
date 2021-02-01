import logging
import os

import numpy as np
import torch
import ase
from ase.db import connect
from schnetpack2.environment import SimpleEnvironmentProvider, collect_atom_triples
from schnetpack2.custom.environment import collect_atom_triples_offsets
from schnetpack2.data import AtomsData, Structure
from torch.utils.data import Dataset, DataLoader
import pdb

cuda = torch.cuda.is_available()

class RandomCrystal(AtomsData):
    def __init__(self, num_elements, len_dataset, max_elements=90, max_num_atoms=50, subset=None, properties=[], environment_provider=SimpleEnvironmentProvider(),
                 collect_triples=False, center_positions=False):
        self.num_elements  = num_elements
        self.len_dataset   = len_dataset
        self.max_elements  = max_elements
        self.max_num_atoms = max_num_atoms
        
        self.subset = subset
        self.properties = properties
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centered = center_positions
        
    def __len__(self):
        return 10*self.len_dataset
        
    def __getitem__(self, idx):
        at = self.create_random_atom()
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
        
    def create_random_atom(self):
#        pdb.set_trace()
        num_atoms = np.random.randint(low=self.num_elements, high=self.max_num_atoms, size = None)
        
        elements   = np.random.choice(self.max_elements, size=self.num_elements, replace=False)
        at_numbers = np.hstack([elements, np.random.choice(elements, size=num_atoms-len(elements), replace=True)])
        
        cell = 3*(num_atoms)**(1/3)*(np.eye(3) + np.random.rand(3,3)/2)
        
        positions = np.random.rand(num_atoms,3)
        
        return ase.Atoms(positions = positions.dot(cell), numbers = at_numbers, cell=cell, pbc=3*[True])
    
