import sys
#sys.path.insert(0,r'C:\Users\Michael\OneDrive\schnetgit\schnetpack\src')
import fastai

print(fastai.__version__)
print(fastai.__file__)

from schnetpack2.custom.fastai.basic_data import DataBunch
from fastai.basic_train import Learner

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
#from schnetpack2.custom.Stepper_fastai1 import Stepper_own

#
import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
from schnetpack2.custom.datasets.ICSD import ICSD
import pdb
import schnetpack2.custom.representation
#from schnetpack2.custom.loss import MSEloss, logrootloss
from schnetpack2.custom.metrics import Emetric,Fmetric


#state = torch.load(os.path.join(args.modelpath,'msenergyforce16large2.h5'))
#model.load_state_dict(state,strict=False)
#from fastai.callbacks.fp16 import *
from schnetpack2.custom.fastai.basic_train import Learner, LearnerCallback
from schnetpack2.custom.fastai.train import *
#from schnetpack2.custom.fastai.callbacks.hooks import SaveAllGrads
from schnetpack2.custom.fastai.torch_core import MetricsList

from schnetpack2.train.hooks import Hook

from schnetpack2.custom.optimizers import SGD_sdr, Adam_sdr
from schnetpack2.custom.nn.cutoff import CosineCutoff

#from schnetpack2.custom.atomistic import TripletWise, TripletWise2
import schnetpack2.custom.atomistic

from schnetpack2.custom.triplets.Losses import TripletLoss, SemihardNegativeTripletSelector, OnlineTripletLoss
from schnetpack2.custom.triplets.Losses_triplets import semihard_negative
from schnetpack2.custom.triplets.Dataloaders import TripletsOfflineDataset, TripletsOnlineDataset
from schnetpack2.custom.triplets.Callbacks import TripletSetter
from schnetpack2.custom.triplets.Transforms import MutateAtom, MutateAtomOnline
from schnetpack2.custom.triplets.Samplers import SequentialNRepeatSampler, RandomSequentialNRepeatSampler, BalancedBatchSamplerReplace

import matplotlib.pyplot as plt

from schnetpack2.custom.utils.save_embeddings import save_embeddings_triplets

from schnetpack2.custom.utils.model_summary import summary

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

import pdb

#class SC(nn.Module):
#     def __init__(self, n_interactions):
#        super(SC, self).__init__()
#        if not isinstance(n_interactions, list): n_interactions = [n_interactions]
#        self.linear = nn.Linear(sum(n_interactions)+1,1)
#        
#     def reset_parameters(self):
#        print("parameters are reset correctly")
#        self.linear.data = torch.ones_like(self.linear.data)
#
#     def forward(self, x):
#        x = torch.stack(x[1], dim=-1)
#        x = x/(x.pow(2).sum(-1,keepdim=True)+1e-7).sqrt()
#        x = self.linear(x)
#        return x.squeeze()
#        
#class SC_model(nn.Module):
#    def __init__(self,model, n_interactions):
#        super(SC_model, self).__init__()
#        self.model   = model
#        self.SClayer = SC(n_interactions)
#    
#    def forward(self, x):
#        x = self.model(x)
#        return self.SClayer(x)
     

def MSEloss(result, batch, kf=0.001,ke=1):
    ediff = batch['energy'] - result['y']
    ediff = ediff ** 2
    fdiff = batch['forces'] - result['dydx']
    fdiff = fdiff ** 2
    fmean = torch.mean(torch.sum(fdiff,[1,2]))
    emean = torch.mean(3*ediff)
    diff = torch.sqrt(kf*fmean + ke*emean + 1e-5)#+kef*fmean*emean
    return diff

class SaveAllGrads(Hook):
    def __init__(self):
      self.all_grads_dict = {}

    def on_batch_begin(self, model, train_batch):
  #    pdb.set_trace()
      for name, param in model.named_parameters():
        if param.requires_grad:
          param.register_hook(self.save_grad(name))
        if not param.requires_grad:
          print(name)

    def save_grad(self, name):
      def hook(grad):
          self.all_grads_dict[name] = grad
      return hook


def check_for_nans(data_loader):
    data_iter = iter(data_loader)
    for ind, input in enumerate(data_iter):
        for dct in input:
            for k, v in dct.items():
                if (v!=v).any():
                    print("batch {0} nan in {1}".format(ind,k))
                    print(v[np.isnan(v)])
                    raise ValueError('A NaN was found in the data.')
    print("No NaNs found")
                    
def check_for_nans_output(data_loader, model, device):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=1e-5)
    data_iter = iter(data_loader)
    hook= SaveAllGrads()
    model.to(device)
    for ind, input in enumerate(data_iter):
        optimizer.zero_grad()
        input[0] = {k:v.to(device) for k,v in input[0].items()}
        input[1] = {k:v.to(device) for k,v in input[1].items()}

        hook.on_batch_begin(model, input[0])
        output=model(input[0])

#        n_atoms = input[0]['_atom_mask'].sum(1)
        loss = MSEloss(output, input[1], kf=0.0,ke=0.9)
        loss.backward()
        optimizer.step()

                    
    print("No NaNs found")

class Args:    
    def __init__(self):
        self.cuda = True
        self.parallel = False
        self.batch_size = 16
        self.property = 'forces'
        self.datapath = ''
        self.modelpath = r'./mp'
        self.seed = 1337
        self.overwrite = False
        self.split_path = None
        self.split = [65000,7000]
        
        
        self.features = 64
        self.interactions = 3
        self.cutoff = 5.
        self.num_gaussians  = 16
        self.start = 0.0
        self.model = 'schnet'
        self.sc=True
        self.sdr=False
        self.filter_network_type="laguerre"
        self.lat_dims = 2
        self.n_permutations = 100
        self.triplets_online = False
        
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

def get_model_spk(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    from schnetpack2.representation.schnet import SchNetCutoffInteraction,SchNetInteraction
    if args.model == 'schnet':           
        representation = spk.representation.SchNet(
            n_atom_basis=args.features, 
            n_filters=args.features, 
            n_interactions=args.interactions, 
            cutoff=args.cutoff, 
            n_gaussians=args.num_gaussians,
            normalize_filter=False, 
            coupled_interactions=False,
            return_intermediate=False, 
            max_z=100, 
            trainable_gaussians=False)
#        atomwise_output = spk.atomistic.Atomwise(args.features, pool_mode='avg', mean=mean, stddev=stddev,
#                                                 atomref=atomref, train_embeddings=True)

        atomwise_output = spk.custom.atomistic.TripletWise(n_in=args.features, n_out=2, aggregation_mode='sum', n_layers=2, n_neurons=None, outnet=None,
                                                               activation=spk.nn.activations.shifted_softplus, requires_dr=False, return_contributions=False)
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model

def get_model(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    from schnetpack2.custom.representation.schnet import SchNetCutoffInteraction,SchNetInteraction
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
            max_z=100, 
            trainable_gaussians=False,
            distance_expansion=None,
            sc=args.sc,
            start = args.start,
            sdr=args.sdr,
            filter_network_type=args.filter_network_type,
            cutoff_network = CosineCutoff)

        atomwise_output = schnetpack2.custom.atomistic.MultiOutput(args.features, n_out = args.lat_dims, aggregation_mode='avg', mean=mean, stddev=stddev,
                                                 atomref=atomref, train_embeddings=True)
#        atomwise_output = spk.custom.atomistic.TripletWise(n_in=args.features, n_out=2, aggregation_mode='sum', n_layers=2, n_neurons=None, outnet=None,
#                                                               activation=spk.nn.activations.shifted_softplus, requires_dr=False, return_contributions=False)#spk.atomistic.Energy(
#        atomwise_output = spk.custom.atomistic.TripletWise2(args.features, n_out = args.lat_dims,  aggregation_mode='avg', mean=mean, stddev=stddev,
#                                                 atomref=atomref, train_embeddings=True)
#        sc = SC_model(representation, args.interactions)
        
        model = spk.custom.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    #if parallelize:
    #state = torch.load('./GebulkV2/night9.h5')
    
#    model = nn.DataParallel(model)
    #model.load_state_dict(state)
    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model
    
def plot_recorder(recorder, skip_start:int=10, skip_end:int=5):
        "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
        lrs = recorder.lrs[skip_start:-skip_end] if skip_end > 0 else recorder.lrs[skip_start:]
        losses = recorder.losses[skip_start:-skip_end] if skip_end > 0 else recorder.losses[skip_start:]
        np.save("losses", np.array([x.item() for x in losses]))
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
        fig.savefig('lr_finder.png')
        
def test_dataloader():
    ts = TripletSetter(model.cuda(), train_loader, train_loader2, semihard_negative, margin=1.0, triplets_per_class=500)
    ts.on_epoch_begin()
    dl_iter = iter(train_loader)
    
    for image, label in dl_iter:
        print(image)
        
def _schnet_split(m):
    return (m.representation, m.output_modules)

args = Args()


print(torch.cuda.device_count())
torch.cuda.empty_cache()

device = torch.device("cuda")
train_args = args
spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)

path = r'/dev/shm/data/'
data_set = ICSD(path, args.cutoff)

idxs = np.arange(len(data_set))
np.random.shuffle(idxs)
num_train = 4*len(idxs)//5

splits_train = idxs[:num_train]
splits_val   = idxs[num_train:]

data_train = data_set.create_subset(splits_train)
data_val   = data_set.create_subset(splits_val)

if not os.path.isdir(args.modelpath):
    os.makedirs(args.modelpath)
    
if args.split_path is not None:
    copyfile(args.split_path, split_path)
    
if args.triplets_online:
    transform = MutateAtom()
    
    data_train_triplets = TripletsOfflineDataset(data_train, args.n_permutations, transform=transform)
    data_val_triplets   = TripletsOfflineDataset(data_val  , args.n_permutations, transform=transform)

    train_loader = schnetpack2.custom.data.AtomsLoader(data_train_triplets, batch_size=args.batch_size, sampler=RandomSampler(data_train_triplets), num_workers=9*torch.cuda.device_count(), pin_memory=True)#9*torch.cuda.device_count()
                                        
    val_loader   = schnetpack2.custom.data.AtomsLoader(data_val_triplets  , batch_size=args.batch_size, sampler=RandomSampler(data_val_triplets)  , num_workers=9*torch.cuda.device_count(), pin_memory=True)
    
    train_loader2 = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=SequentialNRepeatSampler(data_train,args.n_permutations),
                                        num_workers=9*torch.cuda.device_count(), pin_memory=True)#9*torch.cuda.device_count()
                                        
    val_loader2   = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, sampler=SequentialNRepeatSampler(data_val, args.n_permutations),
                                        num_workers=9*torch.cuda.device_count(), pin_memory=True)
                                        
else:
    transform = MutateAtomOnline()
    
    n_classes = 10
    n_samples = 10
    N_repeats = 20
    
    data_train_triplets = TripletsOnlineDataset(data_train, transform=transform)
    data_val_triplets   = TripletsOnlineDataset(data_val  , transform=transform)
    
    train_loader = schnetpack2.custom.data.AtomsLoader(data_train_triplets, batch_sampler=BalancedBatchSamplerReplace(data_train_triplets, n_classes, n_samples, N_repeats = N_repeats), num_workers=9*torch.cuda.device_count(), pin_memory=True)
                                        
    val_loader   = schnetpack2.custom.data.AtomsLoader(data_val_triplets  , batch_sampler=BalancedBatchSamplerReplace(data_val_triplets  , n_classes, n_samples, N_repeats = N_repeats), num_workers=9*torch.cuda.device_count(), pin_memory=True)
        
#train_loader2 = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)
#val_loader2 = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)
#mean, stddev = train_loader.get_statistics('energy', False)
mean = torch.tensor([-1.5115]) 
stddev = torch.tensor([1.2643])
print(mean,stddev)
#block = SchNetCutoffInteraction(args.features, args.features, args.features, args.cutoff,normalize_filter=False)

model = get_model(train_args, atomref=None, mean=torch.FloatTensor([mean]), stddev=torch.FloatTensor([stddev]),
                  train_loader=train_loader,
                  parallelize=args.parallel)

#test_dataloader()

#check_for_nans(train_loader)
#check_for_nans_output(train_loader, model, device)

#pdb.set_trace()
        
data = DataBunch(train_loader, val_loader,collate_fn=schnetpack2.custom.data.collate_atoms)
                    
learn = Learner(data, model, model_dir=args.modelpath)#,callback_fns=PlotSaveGraph)#ShowGraph)#,callback_fns=SaveAllGrads

#learn.load("/scratch/leuven/412/vsc41276/mp/CNN_16epochs", strict=False)

learn.split(_schnet_split)

learn.freeze()

summary(model, device)

if args.sdr:
    learn.opt_func = Adam_sdr#partial(SGD_sdr, weight_decay=0)#
else:
    learn.opt_func = Adam
    
if args.triplets_online:    
    learn.loss_func = TripletLoss(margin=1.0)

    learn.callbacks.append(TripletSetter(model, train_loader, train_loader2, semihard_negative, margin=1.0, triplets_per_class=100))
    learn.callbacks.append(TripletSetter(model, val_loader  , val_loader2  , semihard_negative, margin=1.0, triplets_per_class=125))
else:
    learn.loss_func = OnlineTripletLoss(1.0 ,SemihardNegativeTripletSelector(margin=1.0))

#learn.lr_find(start_lr=1e-6,end_lr=5e-2,no_grad_val = False,num_it=300)
#plot_recorder(learn.recorder)

plt.show()

torch.cuda.empty_cache()
learn.fit_one_cycle(cyc_len=1, max_lr=1e-2, div_factor=20.0, pct_start=0.25,no_grad_val=False) #,wd=#1e-7#,moms=(0.95, 0.85)

#learn.save('16epochs_triplets')

train_loader   = schnetpack2.custom.data.AtomsLoader(data_train_triplets, batch_size=16, sampler =SequentialNRepeatSampler(data_train_triplets, N_repeats = 500), num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader   = schnetpack2.custom.data.AtomsLoader(data_val_triplets    , batch_size=16, sampler =SequentialNRepeatSampler(data_val_triplets, N_repeats = 500), num_workers=9*torch.cuda.device_count(), pin_memory=True)

model = learn.model

model.eval()

if args.triplets_online:
    pass
else:
    save_embeddings_triplets(train_loader, model, "embeddings_triplets_ICSD_online_train_{}".format(args.lat_dims))
    save_embeddings_triplets(val_loader  , model, "embeddings_triplets_ICSD_online_val_{}".format(args.lat_dims))
#learn.sched.plot()

#learn.loss_func = partial(MSEloss,kf=1,ke=0.01)
#learn.lr_find(start_lr=1e-10,end_lr=1e-4,no_grad_val = False,num_it=300)

#learn.recorder.plot()

#learn.fit_one_cycle(cyc_len=6, max_lr=1e-6,moms=(0.95, 0.85), div_factor=10.0, pct_start=0.1, wd=1e-7,no_grad_val=False)
