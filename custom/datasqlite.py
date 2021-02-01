'''
This module contains all functionalities required to load atomistic data efficiently.

'''
import logging
import os
from ase import Atoms
import numpy as np
import torch
from aiosqlite import connect
from schnetpack2.environment import SimpleEnvironmentProvider, collect_atom_triples
from schnetpack2.custom.environment import collect_atom_triples_offsets
from torch.utils.data import Dataset, DataLoader
import pdb

cuda = torch.cuda.is_available()

def toatoms(self, attach_calculator=False,
                add_additional_information=False):
        """Create Atoms object."""
        atoms = Atoms(self.numbers,
                      self.positions,
                      cell=self.cell,
                      pbc=self.pbc,
                      magmoms=self.get('initial_magmoms'),
                      charges=self.get('initial_charges'),
                      tags=self.get('tags'),
                      masses=self.get('masses'),
                      momenta=self.get('momenta'),
                      constraint=self.constraints)

        if attach_calculator:
            params = self.get('calculator_parameters', {})
            atoms.calc = get_calculator_class(self.calculator)(**params)
        else:
            results = {}
            for prop in all_properties:
                if prop in self:
                    results[prop] = self[prop]
            if results:
                atoms.calc = SinglePointCalculator(atoms, **results)
                atoms.calc.name = self.get('calculator', 'unknown')

        if add_additional_information:
            atoms.info = {}
            atoms.info['unique_id'] = self.unique_id
            if self._keys:
                atoms.info['key_value_pairs'] = self.key_value_pairs
            data = self.get('data')
            if data:
                atoms.info['data'] = data



class AtomsData(Dataset):
    """
    Dataset for atomistic systems and properties

    Args:
        asedb_path (str): path to ASE database
        subset (list): indices of subset (default: None)
        properties (list of str): properties to be loaded
        environment_provider (schnetpack2.environment.EnvironmentProvider):
        pair_provider ():

    """

    def __init__(self, asedb_path, subset=None, properties=[], environment_provider=SimpleEnvironmentProvider,
                 collect_triples = False, pair_provider=None, center_positions=True):
        self.asedb_path = asedb_path
        self.asedb =  connect(asedb_path)
        self.subset = subset
        self.properties = properties
        self.environment_provider = environment_provider
        self.collect_triples = collect_triples
        self.centered = center_positions
        self.item = None

    def set_item(self, item):
        "For inference, will briefly replace the dataset with one that only contains `item`."
        self.item = self.x.process_one(item)
        yield None
        self.item = None

    def create_subset(self, idx):
        idx = np.array(idx)
        subidx = idx if self.subset is None else np.array(self.subset)[idx]
        return AtomsData(self.asedb_path, subidx, self.properties, self.environment_provider, self.pair_provider)

    def __len__(self):
        if self.subset is None:
            return self.asedb.execute('SELECT COUNT(id) FROM systems')
        return len(self.subset)

    def get_atoms(self, idx):
        """
        Return atoms of provided index.

        Args:
            idx (int): atoms index

        Returns:
            ase.Atoms: atoms data

        """
        # get row
        if self.subset is None:
            row =  self.asedb.execute('SELECT * FROM systems where id = ' + str(int(idx) + 1)).fetchone()
        else:
            row =  self.asedb.execute('SELECT * FROM systems where id = ' + str(int(self.subset[idx]) + 1)).fetchone()
        at = row.toatoms()
        return at

    def __getitem__(self, idx):
        # get row
        if self.subset is None:
            row =  self.asedb.execute('SELECT * FROM systems where id = ' + str(int(idx) + 1)).fetchone()
        else:
            row =  self.asedb.execute('SELECT * FROM systems where id = ' + str(int(self.subset[idx]) + 1)).fetchone()
        at = row.toatoms()

        # extract properties
        properties = {}
        for p in self.properties:
            # Capture exception for ISO17 where energies are stored directly in the row
            if p in row:
                prop = row[p]
            else:
                prop = row.data[p]
            try:
                prop.shape
            except AttributeError as e:
                prop = np.array([prop], dtype=np.float32)
            properties[p] = torch.FloatTensor(prop)

               # extract/calculate structure
        properties[Structure.Z] = torch.LongTensor(at.numbers.astype(np.int))

        if 'r_ij' in row.data.keys():
            if row.data['r_ij'] is not None:
                properties[Structure.r_ij] = torch.FloatTensor(row.data['r_ij'].astype(np.float32))
        else:
            properties[Structure.r_ij] = None

        if 'f_ij' in row.data.keys():
            if row.data['f_ij'] is not None:
                properties[Structure.f_ij] = torch.FloatTensor(row.data['f_ij'].astype(np.float32))
        else:
            properties[Structure.f_ij] = None

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
            nbh_idx_j, nbh_idx_k, offsets_j, offsets_k = collect_atom_triples_offsets(nbh_idx, offsets)
            properties[Structure.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
            properties[Structure.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))
            properties[Structure.cell_offset_j]    = torch.FloatTensor(offsets_j.astype(np.float32))
            properties[Structure.cell_offset_k]    = torch.FloatTensor(offsets_k.astype(np.float32))
            
        return properties

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        if split_file is not None and os.path.exists(split_file):
            S = np.load(split_file)
            train_idx = S['train_idx'].tolist()
            val_idx = S['val_idx'].tolist()
            test_idx = S['test_idx'].tolist()
        else:
            if num_train is None or num_val is None:
                raise ValueError(
                    'You have to supply either split sizes (num_train / num_val) or an npz file with splits.')

            idx = np.random.permutation(len(self))
            train_idx = idx[:num_train].tolist()
            val_idx = idx[num_train:num_train + num_val].tolist()
            test_idx = idx[num_train + num_val:].tolist()

            if split_file is not None:
                np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        train = self.create_subset(train_idx)
        val = self.create_subset(val_idx)
        test = self.create_subset(test_idx)
        return train, val, test


class StatisticsAccumulator:

    def __init__(self, batch=False, atomistic=False):
        """
        Use the incremental Welford algorithm described in [1]_ to accumulate
        the mean and standard deviation over a set of samples.

        Args:
            batch: If set to true, assumes sample is batch and uses leading
                   dimension as batch size
            atomistic: If set to true, average over atom dimension

        References:
        -----------
        .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        """
        # Initialize state variables
        self.count = 0  # Sample count
#        self.mean = 0  # Incremental average
#        self.M2 = 0  # Sum of squares of differences
        self.batch = batch
        self.atomistic = atomistic

    def add_sample(self, sample_value, n_atoms_masked=None):
        """
        Add a sample to the accumulator and update running estimators. Differentiates between different types of samples

        Args:
            sample_value: Torch variable/tensor
        """
        # Check different cases
        if not self.batch and not self.atomistic:
            self._add_sample(sample_value)
        elif not self.batch and self.atomistic:
            n_atoms = sample_value.size(0)
            for i in range(n_atoms):
                self._add_sample(sample_value[i, :])
        elif self.batch and not self.atomistic:
            n_batch = sample_value.size(0)
            for i in range(n_batch):
                self._add_sample(sample_value[i, :])
        else:
            n_batch = sample_value.shape[0]
            n_atoms = sample_value.shape[1]
            for i in range(n_batch):
                if len(sample_value.shape) == 3:
                    for i in range(n_batch):
                        for j in range(n_atoms_masked[i].item()):
                            self._add_sample(sample_value[i, j, :])
                elif len(sample_value.shape) == 2:
                    for j in range(n_atoms_masked[i].item()):
                        self._add_sample(sample_value[i, j, :])

    def _add_sample(self, sample_value):
        """
        Add a sample to the accumulator and update running estimators.

        Args:
            sample_value: sample of data
        """
#        if len(sample_value.shape)==2: sample_value = sample_value.sum(0)
         
        if not hasattr(self, 'mean'): 
            if len(sample_value.shape)==1: self.mean = torch.zeros((sample_value.shape[-1]))
            elif len(sample_value.shape)==2: self.mean = torch.zeros((1,sample_value.shape[-1]))
        if not hasattr(self, 'M2'):   
            if len(sample_value.shape)==1: self.M2 = torch.zeros((sample_value.shape[-1]))
            elif len(sample_value.shape)==2: self.M2 = torch.zeros((1,sample_value.shape[-1]))
        
        # Update count
        self.count += 1
        
        delta_old = sample_value - self.mean
        # Difference to old mean
        self.mean += delta_old / self.count
        # Update mean estimate
        delta_new = sample_value - self.mean
        # Update sum of differences
        self.M2 += delta_old * delta_new

            
    def get_statistics(self):
        """
        Compute statistics of all data collected by the accumulator.

        Returns:
            mean:   Mean of data
            stddev: Standard deviation of data
        """
        # Compute standard deviation from M2
        mean = self.mean
        stddev = np.sqrt(self.M2 / self.count)
#        stddev = self.stddev
        # TODO: Should no longer be necessary
        # Convert to torch arrays
        # if type(self.mean) == np.ndarray:
        #    mean = torch.FloatTensor(self.mean)
        #    stddev = torch.FloatTensor(stddev)
        # else:
        #    mean = torch.FloatTensor([self.mean])
        #    stddev = torch.FloatTensor([stddev])
        return mean, stddev


def collate_atoms(examples):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        batch: mini-batch of atomistic systems
    """

    properties = {k : v for k,v in examples[0].items() if v is not None}
    keys = properties.keys()

    # initialize maximum sizes
    max_size = {
        prop: np.array(val.size(), dtype=np.int)
        for prop, val in properties.items()
    }

    # get maximum sizes
    for properties in examples[1:]:
        for k in keys:
            max_size[k] = np.maximum(max_size[k], np.array(properties[k].size(), dtype=np.int))

    # initialize batch
    batch = {
        p: torch.zeros(len(examples), *[int(ss) for ss in size]).type(examples[0][p].type()) for p, size in
        max_size.items()
    }
    batch[Structure.neighbor_mask] = torch.zeros_like(batch[Structure.neighbors]).float()
    batch[Structure.atom_mask] = torch.zeros_like(batch[Structure.Z]).float()

    # If neighbor pairs are requested, construct mask placeholders
    # Since the structure of both idx_j and idx_k is identical
    # (not the values), only one cutoff mask has to be generated
    if Structure.neighbor_pairs_j in properties:
        batch[Structure.neighbor_pairs_mask] = torch.zeros_like(batch[Structure.neighbor_pairs_j]).float()

    # build batch and pad
    for k, properties in enumerate(examples):
        for prop, val in properties.items():
            if val is None:
                continue
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val

        # add mask
        nbh = properties[Structure.neighbors]
        shape = nbh.size()
        s = (k,) + tuple([slice(0, d) for d in shape])
        mask = nbh >= 0
        batch[Structure.neighbor_mask][s] = mask

        batch[Structure.neighbors][s] = nbh * mask.long()

        z = properties[Structure.Z]
        shape = z.size()
        s = (k,) + tuple([slice(0, d) for d in shape])
        batch[Structure.atom_mask][s] = z > 0

        # Check if neighbor pair indices are present
        # Since the structure of both idx_j and idx_k is identical
        # (not the values), only one cutoff mask has to be generated
        if Structure.neighbor_pairs_j in properties:
            nbh_idx_j = properties[Structure.neighbor_pairs_j]
            shape = nbh_idx_j.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            mask_pairs = nbh_idx_j >= 0
            batch[Structure.neighbor_pairs_mask][s] = mask_pairs
            batch[Structure.neighbor_pairs_j][s] = nbh_idx_j * mask_pairs.long()
            batch[Structure.neighbor_pairs_k][s] = properties[Structure.neighbor_pairs_k] * mask_pairs.long()

    # wrap everything in variables
#    input_keys = ['_atomic_numbers', '_atom_mask', '_positions', '_cell', '_neighbors', '_neighbor_mask',
#                '_cell_offset', '_cell_offset_j', '_cell_offset_k', '_neighbor_pairs_j', '_neighbor_pairs_k', '_neighbor_pairs_mask',  '_idx','_r_ij','_f_ij']
#    output_keys = [k for k in properties.keys() if k not in input_keys]
#    
#    input_keys_compact = ['_atomic_numbers', '_positions', '_cell', '_cell_offset', '_cell_offset_j', '_cell_offset_k', '_neighbors', '_neighbor_mask', '_atom_mask','_r_ij','_f_ij']
#
#    inputs  = {k: v for k, v in batch.items() if k in input_keys_compact}
#    outputs = {k: v for k, v in batch.items() if k in output_keys}    
    
    return batch#, outputs

def prepare_atom(at, env, centered=False, collect_triples=False):    
    properties = {}

    positions = at.positions.astype(np.float32)
    if centered:
        positions -= at.get_center_of_mass()
    properties[Structure.R] = torch.FloatTensor(positions)

    properties[Structure.cell] = torch.FloatTensor(at.cell.astype(np.float32))

    # get atom environment
    nbh_idx, offsets = env.get_environment(0, at)

    properties[Structure.neighbors] = torch.LongTensor(nbh_idx.astype(np.int))
    properties[Structure.cell_offset] = torch.FloatTensor(offsets.astype(np.float32))
    properties['_idx'] = torch.LongTensor(np.array([0], dtype=np.int))

    if collect_triples:
        nbh_idx_j, nbh_idx_k, offsets_j, offsets_k = collect_atom_triples_offsets(nbh_idx, offsets)
        properties[Structure.neighbor_pairs_j] = torch.LongTensor(nbh_idx_j.astype(np.int))
        properties[Structure.neighbor_pairs_k] = torch.LongTensor(nbh_idx_k.astype(np.int))
        properties[Structure.cell_offset_j]    = torch.FloatTensor(offsets_j.astype(np.float32))
        properties[Structure.cell_offset_k]    = torch.FloatTensor(offsets_k.astype(np.float32))
        
    return properties


class AtomsLoader(DataLoader):
    r"""
    Convenience for ``torch.data.DataLoader`` which already uses the correct collate_fn for AtomsData and
    provides functionality for calculating mean and stddev.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a mini-batch (default: collate_atons).
        pin_memory (bool, optional): If ``True``, the data loader will copy tensors
            into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: None)

    """

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=collate_atoms, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(AtomsLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                          num_workers, collate_fn, pin_memory, drop_last,
                                          timeout, worker_init_fn)


#    def get_batch(self, indices):
#        res = self.collate_fn([self.dataset[i] for i in indices])
#        return res
#
#    def __iter__(self):
#        if self.num_workers==0:
#            for batch in map(self.get_batch, iter(self.batch_sampler)):
#                yield get_tensor(batch, self.pin_memory, self.half)
#        else:
#            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
#                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
#                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
#                    for batch in e.map(self.get_batch, c):
#                        yield get_tensor(batch, self.pin_memory, self.half)

    def get_statistics(self, property_name, atomistic=False, atomref=None, split_file=None):
        """
        Compute mean and variance of a property.
        Uses the incremental Welford algorithm implemented in StatisticsAccumulator

        Args:
            property_name (str):  Name of the property for which the mean and standard
                            deviation should be computed
            atomistic (bool): If set to true, averages over atoms
            atomref (np.ndarray): atomref (default: None)
            split_file (str): path to split file. If specified, mean and std will be cached in this file (default: None)

        Returns:
            mean:           Mean value
            stddev:         Standard deviation

        """
        with torch.no_grad():
            calc_stats = True
            if split_file is not None and os.path.exists(split_file):
                split_data = np.load(split_file)

                if 'mean' in split_data and 'stddev' in split_data:
                    mean = torch.from_numpy(split_data['mean'])
                    stddev = torch.from_numpy(split_data['stddev'])
                    calc_stats = False
                    logging.info("cached statistics was loaded...")

            if calc_stats:
                statistics = StatisticsAccumulator(batch=True, atomistic=atomistic)
                logging.info("statistics will be calculated...")

                count = 0
                for row in self:
                    self._update_statistic(atomistic, atomref, property_name, row, statistics)
                    count += 1

                mean, stddev = statistics.get_statistics()

                # cache result in split file
                if split_file is not None and os.path.exists(split_file):
                    split_data = np.load(split_file)
                    np.savez(split_file, train_idx=split_data['train_idx'],
                             val_idx=split_data['val_idx'], test_idx=split_data['test_idx'], mean=mean, stddev=stddev)

            return mean, stddev

    def _update_statistic(self, atomistic, atomref, property_name, row, statistics):
        """
        Helper function to update iterative mean / stddev statistics computation
        """
        #print(row)
        #row[property_name]
        if isinstance(row, list): property_value = row[1][property_name]
        if isinstance(row, dict): property_value = row[property_name]
        
        n_atoms = None
        
        if atomref is not None:
            z = row[0]['_atomic_numbers']
            p0 = torch.sum(torch.from_numpy(atomref[z]).float(), dim=1)
            property_value -= p0
        if atomistic:
            n_atoms = torch.sum(row['_atom_mask'], dim=1, keepdim=True)[:,None]
#            if isinstance(row, list): property_value /= n_atoms
#            if isinstance(row, dict): property_value /= n_atoms
            n_atoms = torch.flatten(n_atoms.long())
        statistics.add_sample(property_value, n_atoms)


class Structure:
    Z = '_atomic_numbers'
    atom_mask = '_atom_mask'
    R = '_positions'
    cell = '_cell'
    neighbors = '_neighbors'
    neighbor_mask = '_neighbor_mask'
    cell_offset   = '_cell_offset'
    cell_offset_j = '_cell_offset_j'
    cell_offset_k = '_cell_offset_k'
    neighbor_pairs_j = '_neighbor_pairs_j'
    neighbor_pairs_k = '_neighbor_pairs_k'
    neighbor_pairs_mask = '_neighbor_pairs_mask'
    r_ij = '_r_ij'
    f_ij = '_f_ij'

