#/usr/bin/env python
import schnetpack2.custom.fastai
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
from schnetpack2.custom.loss import MSEloss, logrootloss,MAEloss, PhysNetLoss, NLLMSEloss, NLLMSEloss_forces
from schnetpack2.custom.metrics import Emetric,Fmetric, DipoleMetric, ChargeMetric, CouplingMetric, CouplingMetricKaggle
import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.datasets.kaggle
import schnetpack2.custom.representation
from schnetpack2.custom.GP import GPRegressionLayer, DKLModel, FlattenedMLL, Energy, AtomisticModel
import gpytorch
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

import pdb

class Args:
    def __init__(self):
        self.cuda = True
    
print(torch.cuda.device_count())
args = Args()
args.cuda = True
args.parallel = False
args.batch_size = int(sys.argv[1])*torch.cuda.device_count()
args.property = 'forces'
args.datapath = ''
args.modelpath = os.environ['VSC_SCRATCH']
args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]


args.features = 32
args.interactions = 5#[2,2,2,3,3,3]
args.cutoff = 5.
args.num_gaussians  = 32
args.n_expansion_coeffs = 32
args.start = 0.0
args.model = 'schnet'
args.sc=True
args.maxz = 100
args.outlayers = 5
args.num_inducing_points = 100
args.sdr = False
args.uncertainty = False
args.uncertainty_forces = False
args.maxz = 100
args.outlayers = 5
#args.p = float(sys.argv[5])
#args.var_coeff=float(sys.argv[6])

DEVICE = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
print(DEVICE)
torch.cuda.set_device(DEVICE)

def train(epoch):
    model.train()
    mll.train()

    train_loss = 0.
    for batch_idx, data in enumerate(train_loader):
        data = {key:val.cuda() for key, val in data.items()}
        optimizer.zero_grad()
        output = model(data)
        loss = -mll(output, data)
        loss.backward()
        train_loss += loss.item()/len(output["y"])
        optimizer.step()
            
    print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), train_loss/len(train_loader)))

def test():
    model.eval()
    mll.eval()

    correct = 0
    loss = 0
    for data in val_loader:
        data = {key:val.cuda() for key, val in data.items()}
        output = model(data)
        correct += torch.mean(torch.abs(output["y_dist"].mean - data["energy"]))
        loss += -mll(output, data).item()/len(data["energy"])
    print('Test set: Loss: {}, (MAE: {})'.format(
        loss / float(len(val_loader)), correct / float(len(val_loader))
    ))

def get_model(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    if args.model == 'schnet':
        representation = spk.custom.representation.schnet.SchNet(
            n_atom_basis=args.features, 
            n_filters=args.features, 
            n_interactions=args.interactions, 
            cutoff=args.cutoff, 
            n_gaussians=args.num_gaussians,
            normalize_filter=False, 
            coupled_interactions=False,
            max_z=args.maxz, 
            cutoff_network=schnetpack2.nn.cutoff.CosineCutoff, 
            filter_network_type="original",
            n_expansion_coeffs=args.n_expansion_coeffs,
            trainable_gaussians=False,
            distance_expansion=None,
            sc=args.sc,
            start = args.start,
            bn=False,p=0,debug=False, attention=0)
        
        data_iter = iter(train_loader)
        
        features = []
        
        for ind, x in enumerate(data_iter):
           if ind<30:
#               mask = x["_atom_mask"].byte()[...,None].expand(*x["_atom_mask"].shape,args.features)
               features.append((representation(x)*x["_atom_mask"][...,None]).mean(1))
           else:
               break

        features = torch.cat(features,dim=0)

#        atomwise_output = spk.custom.atomistic.Energy(n_in=args.features, mean=mean, aggregation_mode='sum', stddev=stddev, outnet=DKLModel(features), num_inducing_points = args.num_inducing_points, atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, bn=False,p=0.0)
        atomwise_output = spk.custom.atomistic.Energy(n_in=args.features, mean=mean, aggregation_mode='sum', stddev=stddev, outnet=None, num_inducing_points = args.num_inducing_points, atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, bn=False,p=0.0)

        model = spk.custom.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    #if parallelize:
    #state = torch.load('./bulkV/64epochs.pth')
    
#    model = nn.DataParallel(model)
    #model.load_state_dict(state,strict=False)
    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model.cuda()
    
device = torch.device("cuda")
train_args = args
#spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)
#name = r'/dev/shm/data/train' + sys.argv[1]
#name = r'/dev/shm/data/bulkVtrain3200_'+ sys.argv[1]
name = r'/dev/shm/data/bulkVtrain3200'
data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
#name = r'/dev/shm/data/test' + sys.argv[1]
#name = r'/dev/shm/data/bulkVtest3200_'+ sys.argv[1]
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
val_loader = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)
mean, stddev = train_loader.get_statistics('energy', False)

mean_forces, stddev_forces = train_loader.get_statistics('forces', True)
#mean = -4178.7568
#stddev = 29.6958
if args.uncertainty:
    mean, stddev = torch.tensor([-1202.6432,0]), torch.tensor([12.3304,1])
else:
    mean, stddev = torch.tensor([-1202.6432]), torch.tensor([12.3304])

print(mean,stddev)
print(mean_forces,stddev_forces)

model = get_model(train_args, atomref=None, mean=torch.FloatTensor([mean]), stddev=torch.FloatTensor([stddev]),
                  train_loader=train_loader,
                  parallelize=args.parallel)
data = DataBunch( train_loader, val_loader,collate_fn=schnetpack2.data.collate_atoms)

learn = Learner(data, model, model_dir=args.modelpath)

#pdb.set_trace()   
learn.opt_func = Adam
#learn.loss_func = partial(NLLMSEloss_forces, mean=mean.cuda(), std=stddev.cuda(), mean_forces=mean_forces.cuda(), std_forces=stddev_forces.cuda(), kf=0.001,ke=1)

if args.num_inducing_points:
    learn.loss_func = partial(NLLMSEloss, mean=mean.cuda(), std=stddev.cuda(), kf=0.5,ke=1.0, kgamma = 0.0001)#partial(MSEloss, kf=0.1,ke=1,kef=0)
else:
    learn.loss_func = partial(MSEloss, kf=0.1,ke=1,kef=0)
#optimizer = Adam([
#    {'params': model.representation.parameters(), 'lr': 1e-2},
#    {'params': model.output_modules.parameters(), 'lr': 1e-1},
#])
#likelihood = gpytorch.likelihoods.GaussianLikelihood()
#mll = FlattenedMLL(model.output_modules.out_net.gp_layer, num_data=len(data_train), kf=0.0).train()
#model.train()
learn.metrics=[Emetric,Fmetric] #DipoleMetric, ChargeMetric, CouplingMetric, CouplingMetricKaggle]

#for epoch in range(1, int(sys.argv[2]) + 1):
#    with gpytorch.settings.use_toeplitz(True):
#        train(epoch)
#        test()

#learn.load('trainbulkVsurf128epochs')
torch.cuda.empty_cache()
#print(sys.argv)
#with gpytorch.settings.use_toeplitz(True):
learn.fit_one_cycle(cyc_len=int(sys.argv[2]), max_lr=float(sys.argv[3]),moms=(0.95, 0.85), div_factor=500.0, pct_start=0.05, wd=1e-2,no_grad_val=False)
#print(learn.save(os.path.join(args.modelpath,"PhysNet_{0}_{1}_{2}".format(sys.argv[1], sys.argv[2], sys.argv[3])), return_path=True))
#torch.save(model, os.path.join(args.modelpath,"DKL_weights_{0}_{1}_{2}".format(sys.argv[1], sys.argv[2], sys.argv[3])))

#if args.uncertainty_forces:
#    print(learn.save( "bulkVtrain3200"+"_uncertain_forces_"+str(sys.argv[3]), return_path=True))
if args.num_inducing_points:
    print(learn.save( "bulkVtrain3200"+"_induced_"+str(sys.argv[3]), return_path=True))
else:
    print(learn.save( "bulkVtrain3200_"+str(sys.argv[3]), return_path=True))