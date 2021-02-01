#/usr/bin/env python
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
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

import pdb

class Args:
    def __init__(self):
        self.cuda = True
    

args = Args()
args.cuda = True
args.parallel = False
args.batch_size = 8*torch.cuda.device_count()
args.property = 'forces'
args.datapath = ''
args.modelpath = r'/dev/shm/data/models'
args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]


args.features = 256
args.interactions = [2,2,2,3,3,3]
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
            n_expansion_coeffs=args.num_gaussians,
            trainable_gaussians=False,
            distance_expansion=None,
            sc=args.sc,
            start = args.start,
        bn=False,p=0,debug=False, attention=0)
        atomwise_output = spk.custom.atomistic.Energy(args.features, mean=mean, aggregation_mode='sum', stddev=stddev,
                                                atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers,bn=False,p=0.1)
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    #if parallelize:
    #state = torch.load('./bulkV/64epochs.pth')
    
    model = nn.DataParallel(model)
    #model.load_state_dict(state,strict=False)
    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model
device = torch.device("cuda")
train_args = args
spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)
name = r'/dev/shm/data/train' + sys.argv[1]
name = r'/dev/shm/data/bulkVtrain3200'
data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
name = r'/dev/shm/data/test' + sys.argv[1]
name = r'/dev/shm/data/bulkVtest3200'
data_val = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])

if not os.path.isdir(args.modelpath):
    os.makedirs(args.modelpath)
    
if args.split_path is not None:
    copyfile(args.split_path, split_path)
    
#from sklearn.model_selection import train_test_split
#train,test = train_test_split(df, test_size=0.20, random_state=42,stratify=df['Ebin'].values)

train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                    num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)
mean, stddev = train_loader.get_statistics('energy', False)
#mean = -4178.7568
#stddev = 29.6958
print(mean,stddev)

model = get_model(train_args, atomref=None, mean=torch.FloatTensor([mean]), stddev=torch.FloatTensor([stddev]),
                  train_loader=train_loader,
                  parallelize=args.parallel)
data = DataBunch( train_loader, val_loader,collate_fn=schnetpack2.data.collate_atoms)


learn = Learner(data, model, model_dir=args.modelpath)

learn.purge()
learn.opt_func = Adam
learn.loss_func = partial(MSEloss,kf=1.0,ke=0.1)
learn.metrics=[Emetric,Fmetric]
#learn.load('trainbulkVsurf128epochs')
torch.cuda.empty_cache()

import matplotlib.pyplot as plt
import numpy as np

def plot_recorder(recorder, save_name, skip_start:int=10, skip_end:int=5):
       "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
       lrs = recorder.lrs[skip_start:-skip_end] if skip_end > 0 else recorder.lrs[skip_start:]
       losses = recorder.losses[skip_start:-skip_end] if skip_end > 0 else recorder.losses[skip_start:]
       np.save(save_name, np.array([x.item() for x in losses]))
       fig, ax = plt.subplots(1,1)
       ax.plot(lrs, losses)
       ax.set_ylabel("Loss")
       ax.set_xlabel("Learning Rate")
       ax.set_xscale('log')
       ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
       mg = (np.gradient(np.array([x.item() for x in losses]))).argmin()
       print(f"Min numerical gradient: {lrs[mg]:.2E}")
       ax.plot(lrs[mg],losses[mg],markersize=10,marker='o',color='red')
       plt.show()
       fig.savefig(save_name+'.png')


learn.lr_find(start_lr=1e-6,end_lr=1e-1, num_it=300)
import os
os.chdir('/dev/shm/data')
plot_recorder(learn.recorder, sys.argv[1])
