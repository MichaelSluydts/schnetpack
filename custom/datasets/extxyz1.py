from ase.db import connect
from ase.io.extxyz import read_xyz

from schnetpack2.custom.data import AtomsData
from schnetpack2.environment import SimpleEnvironmentProvider

__all__ = ['ExtXYZ', 'parse_extxyz']


def parse_extxyz(dbpath, xyzpath, env, cache=False):
    r"""Parses file in XYZ format and writes content to sqllite database

    Args:
        dbpath(str): path to sqllite database
        xyzpath (str): path to file with xyz file format
    """
    with connect(dbpath, use_lock_file=False) as conn:
        with open(xyzpath) as f:
            atoms = []
            energies = []
            forces = []
            energiesperatom = []
            eform = []
            eformperatom = []
            ehull = []
            ebin = []
            for at in read_xyz(f, index=slice(None)):
                nat = at.get_number_of_atoms()
                energies.append(at.get_total_energy())
                try:
                    force = at.get_forces()
                except:
                    force = 0.*at.get_positions()
                forces.append(force)
                atoms.append(at)
                energiesperatom.append(energies[-1] / nat)
                eform.append(energies[-1] - nat*-19.0329202806)
                eformperatom.append(eform[-1] / nat)
                ehull.append(0)
                ebin.append(0)
            energies = np.array(energies)
            m = np.mean(energies)
            emin = np.min(energies)
            emax = np.max(energies)
            # energies -= m

            for i in range(len(atoms)):
                # atoms[i].energy = energies[i]
                atoms[i]._calc.results['energy'] = energies[i]
                r_ij, f_ij = None, None
                if cache:
                    r_ij, f_ij = neighbor_gen(at, distance_expansion=None, cutoff=5.0, n_gaussians=25,
                                              trainable_gaussians=False,
                                              environment_provider=env, collect_triples=False, pair_provider=None,
                                              center_positions=True)
                conn.write(atoms[i],
                           data={ExtXYZ.E: energies[i], ExtXYZ.F: forces[i], ExtXYZ.E + 'peratom': energiesperatom[i],
                                 'Eform': eform[i], 'Eformperatom': eformperatom[i], 'Ehull': ehull[i], 'Ebin': ebin[i],
                                 'mean': m, 'r_ij': r_ij, 'f_ij': f_ij})


class ExtXYZ(AtomsData):
    '''
    Loader for MD data in extended XYZ format

    :param path: Path to database
    '''

    E = "energy"
    F = "forces"

    def __init__(self, dbpath, xyzpath, subset=None, properties=[], environment_provider=SimpleEnvironmentProvider(),
                 collect_triples=False,
                 pair_provider=None, center_positions=True, cache=False):
        if not os.path.exists(dbpath):
            if not os.path.isdir('/'.join(dbpath.split('/')[:-1])):
                os.makedirs('/'.join(dbpath.split('/')[:-1]))
            parse_extxyz(dbpath, xyzpath, environment_provider, cache)
        super(ExtXYZ, self).__init__(dbpath, subset, properties, environment_provider, collect_triples, pair_provider,
                                     center_positions)


from schnetpack2.custom.data import Structure
from fastai.torch_core import tensor, to_np, to_device
from schnetpack2.environment import ASEEnvironmentProvider, collect_atom_triples
import os
import numpy as np
import schnetpack as spk


def neighbor_gen(at, distance_expansion=None, cutoff=5.0, n_gaussians=25, trainable_gaussians=False,
                 environment_provider=ASEEnvironmentProvider(5.0),
                 collect_triples=False, pair_provider=None, center_positions=True):
    properties = {}
    properties[Structure.Z] = tensor(at.numbers.astype(np.int)).unsqueeze(0)

    positions = at.positions.astype(np.float32)
    if center_positions:
        positions -= at.get_center_of_mass()
    properties[Structure.R] = tensor(positions).unsqueeze(0)

    properties[Structure.cell] = tensor(at.cell.astype(np.float32)).unsqueeze(0)

    # get atom environment
    idx = 0
    nbh_idx, offsets = environment_provider.get_environment(idx, at)

    properties[Structure.neighbors] = tensor(nbh_idx.astype(np.int)).unsqueeze(0)
    properties[Structure.cell_offset] = tensor(offsets.astype(np.float32)).unsqueeze(0)
    properties[Structure.neighbor_mask] = None
    properties['_idx'] = tensor(np.array([idx], dtype=np.int)).unsqueeze(0)

    if collect_triples:
        nbh_idx_j, nbh_idx_k = collect_atom_triples(nbh_idx)
        properties[Structure.neighbor_pairs_j] = tensor(nbh_idx_j.astype(np.int))
        properties[Structure.neighbor_pairs_k] = tensor(nbh_idx_k.astype(np.int))

    model = spk.custom.representation.RBF(distance_expansion=distance_expansion, cutoff=cutoff, n_gaussians=n_gaussians,
                                          trainable_gaussians=trainable_gaussians)
    model = to_device(model)
    r, f = model.forward(properties)
    return to_np(r.squeeze()), to_np(f.squeeze())
