# coding: utf-8
import sys
#import fastai

from schnetpack2.custom.fastai.basic_data import DataBunch
from schnetpack2.custom.fastai.basic_train import Learner

import argparse
import logging
import os
import sys
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
from schnetpack2.custom.representation.schnet import SchNetCutoffInteraction,SchNetInteraction

from schnetpack2.custom.optimizers import SGD_sdr, Adam_sdr

import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.representation
from schnetpack2.custom.loss import MSEloss, logrootloss
from schnetpack2.custom.metrics import Emetric,Fmetric
from schnetpack2.custom.fastai.basic_train import Learner
from schnetpack2.custom.fastai.train import *
#from schnetpack2.custom.fastai.callbacks.hooks import SaveAllGrads
from schnetpack2.custom.fastai.torch_core import flatten_model

import pdb

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

class Args:
    def __init__(self):
        self.cuda = True

args = Args()
args.cuda = True
args.parallel = False
args.batch_size = 4
args.property = 'forces'
args.datapath = ''
args.modelpath = r'./GebulkV'
args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]

args.features = 128
args.interactions = [2 for i in range(32)]
args.cutoff = 5.
args.num_gaussians  = 64
args.start = 0.0
args.model = 'schnet'
args.sc=True
args.maxz = 100
args.outlayers = 5

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

block = SchNetCutoffInteraction(args.features, args.features, args.features, args.cutoff,normalize_filter=False)


def get_model(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    if args.model == 'schnet':
        representation = spk.custom.representation.SchNet(
            n_atom_basis=args.features, 
            n_filters=args.features, 
            n_interactions=args.interactions, 
            cutoff=args.cutoff, 
            n_gaussians=args.num_gaussians,
            normalize_filter=False, 
            coupled_interactions=False,
            return_intermediate=False, 
            max_z=args.maxz, 
            cutoff_network=schnetpack2.nn.cutoff.CosineCutoff,
            filter_network_type="original",
            trainable_gaussians=False,
            distance_expansion=None,
            sc=args.sc,
            start = args.start)
        atomwise_output = spk.custom.atomistic.Energy(args.features, mean=mean, aggregation_mode='sum', stddev=stddev,
                                                atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers)
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

#    state = torch.load('./GebulkV/64epochs.pth')
    
    model = nn.DataParallel(model)
#    model.load_state_dict(state,strict=False)
    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model

def groupness(m):
    return tuple(model.module.output_modules.children())

device = torch.device("cuda")
train_args = args
spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)

#name = r'/dev/shm/data/Gepuresurface100small'
name = r'/dev/shm/data/bulkVtrain3200'
data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
#name = r'/dev/shm/data/Gepuresurface111small'
name = r'/dev/shm/data/bulkVtest3200'
data_val = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])

if not os.path.isdir(args.modelpath):
    os.makedirs(args.modelpath)
    
if args.split_path is not None:
    copyfile(args.split_path, split_path)
    
train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                    num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)
mean, stddev = train_loader.get_statistics('energy', False)

print(mean,stddev)
model = get_model(train_args, atomref=None, mean=torch.FloatTensor([mean]), stddev=torch.FloatTensor([stddev]),
                  train_loader=train_loader,
                  parallelize=args.parallel)

data = DataBunch( train_loader, val_loader,collate_fn=schnetpack2.data.collate_atoms)

learn = Learner(data, model, model_dir=args.modelpath)#,callback_fns=ShowGraph)#,callback_fns=SaveAllGrads

learn.opt_func = Adam
learn.loss_func = partial(MSEloss,kf=1,ke=0.1)

learn.metrics=[Emetric,Fmetric]

learn.fit_one_cycle(cyc_len=2, max_lr=3e-3,moms=(0.95, 0.85), div_factor=100.0, pct_start=0.3, wd=1e-7,no_grad_val=False)

learn.to_fp16(loss_scale=100, max_noskip=1000, dynamic=False, clip=None,flat_master=True)

torch.cuda.empty_cache()

#learn.lr_find(start_lr=1e-6,end_lr=1e-2,no_grad_val = False,num_it=63)
#learn.recorder.plot()

torch.cuda.empty_cache()
learn.fit_one_cycle(cyc_len=16, max_lr=3e-3,moms=(0.95, 0.85), div_factor=100.0, pct_start=0.3, wd=1e-7,no_grad_val=False)

get_ipython().run_line_magic('pinfo2', 'learn.validate')


learn.lr_find(start_lr=1e-6,end_lr=1e-2,no_grad_val = False,num_it=63)
learn.recorder.plot()

learn.fit_one_cycle(cyc_len=8, max_lr=2e-5,moms=(0.95, 0.85), div_factor=10.0, pct_start=0.3, wd=1e-7,no_grad_val=False)

flatten_model(learn.model)

learn.split(groupness)
learn.layer_groups
learn.freeze_to(1)

learn.lr_find(start_lr=1e-6,end_lr=1e-2,no_grad_val = False,num_it=63)
learn.recorder.plot()

learn.fit_one_cycle(cyc_len=16, max_lr=2e-4,moms=(0.95, 0.85), div_factor=400.0, pct_start=0.1, wd=1e-7,no_grad_val=False)

model.summary()

learn.save('surfsmall')
learn.sched.plot()

learn.loss_func = partial(MSEloss,kf=1,ke=0.01)

learn.lr_find(start_lr=1e-10,end_lr=1e-4,no_grad_val = False,num_it=300)
learn.recorder.plot()

learn.fit_one_cycle(cyc_len=6, max_lr=1e-6,moms=(0.95, 0.85), div_factor=10.0, pct_start=0.1, wd=1e-7,no_grad_val=False)
