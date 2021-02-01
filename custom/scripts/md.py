#!/usr/bin/env python
# coding: utf-8

import fastai
from schnetpack2.custom.fastai.basic_data import DataBunch
from schnetpack2.custom.fastai.basic_train import Learner
from schnetpack2.custom.fastai.train import *
import sys,os
import argparse
import logging
from shutil import copyfile, rmtree
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import schnetpack as spk
from torch.optim import Adam, SGD
from torch.utils.data.sampler import RandomSampler
from schnetpack2.environment import ASEEnvironmentProvider
from schnetpack2.datasets import MaterialsProject
from schnetpack2.utils import to_json, read_from_json, compute_params
from schnetpack2.custom.representation.schnet import SchNetInteraction
from schnetpack2.custom.loss import MSEloss, logrootloss,MAEloss
from schnetpack2.custom.metrics import Emetric,Fmetric
import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.representation
import ase.io
import numpy as np
from yaff import *
import h5py as h5
from molmod.periodic import periodic
from schnetpack2.md import AtomsConverter
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
from ase import Atoms
import pdb
from schnetpack2.custom.interface.yaff import ML_FF

class Args:
    def __init__(self):
        self.cuda = True

args = Args()
args.cuda = True
args.parallel = False
args.batch_size = 1
args.property = 'forces'
args.datapath = ''
args.modelpath = r'./GebulkV'
args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]


args.features = 256
args.filters = args.features
args.interactions = [2,2,2,3,3,3]
args.cutoff = 5.
args.num_gaussians  = 64
args.n_expansion_coefficients = args.num_gaussians
args.start = 0.0
args.model = 'schnet'
args.sc=True
args.maxz = 100
args.outlayers = 5
args.NPT=False

print(args.features,args.filters,args.num_gaussians,args.interactions,args.cutoff)


def evaluate(args, model, train_loader, val_loader, test_loader, device):
    header = ['Subset', 'energy MAE', 'forces MAE']

    metrics = [spk.metrics.MeanAbsoluteError('energy', 'y'),
               spk.metrics.MeanAbsoluteError('forces', 'dydx')
               ]

    results = []
    if 'train' in args.split:
        results.append(['training'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, train_loader, device)])

    if 'validation' in args.split:
        results.append(['validation'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, val_loader, device)])

    if 'test' in args.split:
        results.append(['test'] + ['%.5f' % i for i in evaluate_dataset(metrics, model, test_loader, device)])

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results, header=header, fmt='%s', delimiter=',')


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [
        metric.aggregate() for metric in metrics
    ]
    return results

def get_model(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    if args.model == 'schnet':
        representation = spk.custom.representation.SchNet(
            n_atom_basis=args.features,
            n_filters=args.filters,
            n_interactions=args.interactions,
            cutoff=args.cutoff,
            n_gaussians=args.num_gaussians,
            normalize_filter=False,
            coupled_interactions=False,
            return_intermediate=False,
            max_z=args.maxz,
            cutoff_network=schnetpack2.nn.cutoff.CosineCutoff,
            filter_network_type="original",
            n_expansion_coeffs=args.n_expansion_coefficients,
            trainable_gaussians=True,
            distance_expansion=None,
            sc=args.sc,
            start = args.start,
            bn=False,p=0,debug=False, attention=0
            ,return_stress=args.NPT)
        atomwise_output = spk.custom.atomistic.Energy(args.features, mean=mean, aggregation_mode='sum', stddev=stddev,
                                                atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers,bn=False,p=0
                                                      ,return_stress=args.NPT,return_hessian=False)
        model = spk.custom.atomistic.AtomisticModel(representation, atomwise_output)
        
    else:
        raise ValueError('Unsupported model class:', args.model)

    #if parallelize:
#    model = nn.DataParallel(model)
#    state = torch.load('/dev/shm/data/models/all.pth')
#
#    print(set(model.module.state_dict().keys()).difference(state['model'].keys()))
#    pretrained_dict = state['model']
#    model_dict = model.module.state_dict()
#
#    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#    model_dict.update(pretrained_dict) 
#    print(model_dict['output_modules.out_net.1.out_net.4.weight'])
#    model.module.load_state_dict(pretrained_dict,strict=False)
#    print(model.module.state_dict()['output_modules.out_net.1.out_net.4.weight'])
    #model.load_state_dict(state,strict=False)
    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model
device = torch.device("cuda")
train_args = args
#spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)
name = r'/dev/shm/data/bulkVtrain3200'
data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
name = r'/dev/shm/data/bulkVtest3200'
data_val = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])

if not os.path.isdir(args.modelpath):
    os.makedirs(args.modelpath)

if args.split_path is not None:
    copyfile(args.split_path, split_path)

#from sklearn.model_selection import train_test_split
#train,test = train_test_split(df, test_size=0.20, random_state=42,stratify=df['Ebin'].values)
print(args.batch_size)
train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                    num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count())
#mean, stddev = train_loader.get_statistics('energy', False)
mean = -1480.1879
stddev = 884.9210
print(mean,stddev)

model = get_model(train_args, atomref=None, mean=torch.FloatTensor([mean]), stddev=torch.FloatTensor([stddev]),
                  train_loader=train_loader,
                  parallelize=args.parallel)
data = DataBunch( train_loader, val_loader,collate_fn=schnetpack2.data.collate_atoms)


learn = Learner(data, model, model_dir=args.modelpath)


env = ASEEnvironmentProvider(args.cutoff)
conv = schnetpack2.md.AtomsConverter(device=device)

#def atoms2yaff(at):
#    numbers = at.get_atomic_numbers()
#    pos = at.get_positions()*angstrom
#    rvecs = at.get_cell()*angstrom
#    system = System( numbers, pos, rvecs=rvecs)
#    return system
#
#def yaff2atoms(sys):
#    atom = Atoms(numbers=sys.numbers,positions=sys.pos/angstrom,cell=sys.cell.rvecs/angstrom,pbc=True
#)
#    return atom
#
#def atoms2schnet(at):
#    return conv.convert_atoms(atoms=at)
#
#def yaff2schnet(at):
#    return atoms2schnet(yaff2atoms(at))


#class ML_FF(ForcePart):
#    def __init__(self, system, model):
#        ForcePart.__init__(self, 'ml_ff', system)
#        self.system = system
#        self.model = model
#        model.eval()
#
#    def _internal_compute(self, gpos, vtens):
#
#        results = self.model(yaff2schnet(self.system))
#        if 'stress' in results.keys():
#            energy, new_gpos, new_vtens = results['y'], results['dydx'], results['stress']
#        else:
#            energy, new_gpos, new_vtens = results['y'], results['dydx'], 0
#        
#        if not gpos is None:
#            gpos[:, :] = - new_gpos * electronvolt / angstrom
#
#        if not vtens is None: # vtens = F x R
#            vtens[:, :] = new_vtens * electronvolt
#
#        return energy * electronvolt

#def NVE(system, steps, nprint = 10, dt = 1, temp = 600, start = 0, name = 'run',
#        restart = None):
#    ff = ForceField(system, [ML_FF(system)])
#
#    f = h5.File(name + '.h5', mode = 'w')
#    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
#    sl = VerletScreenLog(step = 250)
#    xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
#
#    f2 = h5.File(name + '_restart.h5', mode = 'w')
#    restart_writer = RestartWriter(f2, start = start, step = 5000)
#
#    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [restart_writer, sl, xyz, hdf5_writer], temp0 = temp, restart_h5 = restart)
#    verlet.run(steps)
#
#    f.close()
#
#def NPT(system, model, steps, nprint = 10, dt = 1, temp = 600, start = 0, name = 'run'):
#    ff = ForceField(system, [ML_FF(system, model)])
#
#    thermo = NHCThermostat(temp = temp)
#
#    baro = MTKBarostat(ff, temp = temp, press = 1 * 1e+05 * pascal)
#    tbc = TBCombination(thermo, baro)
#
#    #f = h5.File(name + '.h5', mode = 'w')
#    #hdf5_writer = HDF5Writer(f, start = start, step = nprint)
#    sl = VerletScreenLog(step = 10)
#    xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
#    f2 = h5.File(name + '_restart.h5', mode = 'w')
#    restart_writer = RestartWriter(f2, start = start, step = 5000)
#
#    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [restart_writer, sl, tbc, xyz], temp0 = temp)
#    verlet.run(steps)
#
#    f.close()
#    
#def NVT(system, model, steps, nprint = 10, dt = 1, temp = 600, start = 0, name = 'run'):
#    ff = ForceField(system, [ML_FF(system, model)])
#
#    thermo = NHCThermostat(temp = temp)
#
#    #baro = MTKBarostat(ff, temp = temp, press = 1 * 1e+05 * pascal)
#    #tbc = TBCombination(thermo, baro)
#
#    #f = h5.File(name + '.h5', mode = 'w')
#    #hdf5_writer = HDF5Writer(f, start = start, step = nprint)
#    sl = VerletScreenLog(step = 10)
#    xyz = XYZWriter(name + '.xyz', start = start, step = nprint)
#    f2 = h5.File(name + '_restart.h5', mode = 'w')
#    restart_writer = RestartWriter(f2, start = start, step = 5000)
#
#    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [restart_writer, sl, thermo, xyz], temp0 = temp)
#    verlet.run(steps)
#
#    f2.close()



torch.cuda.empty_cache()
#poscarpath = '/dev/shm/benchmarks/poscars/size/vasprun999.xml'
poscarpath = '/dev/shm/data/vasprun215.xml'

atom = ase.io.read(poscarpath,index=0)

#MD run
#system = atoms2yaff(atom)
steps = 500

#mlff = ML_FF(system,model)
model.eval()
mlff = ML_FF(atom, model, conv, env)

if args.NPT:
  npt = mlff.NPT(steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')
else:
  nvt = mlff.NVT(steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')

#npt = NVT(system, model, steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')
#npt = NVT(system, model, steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')