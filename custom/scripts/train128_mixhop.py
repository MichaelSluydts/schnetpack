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
from schnetpack2.environment import ASEEnvironmentProvider, SimpleEnvironmentProvider, SimpleEnvironmentProviderAuto
from schnetpack2.datasets import MaterialsProject
from schnetpack2.utils import to_json, read_from_json, compute_params
from schnetpack2.custom.representation.schnet_mixhop import SchNetMixHop
from schnetpack2.custom.loss import MSEloss, logrootloss,MAEloss, NLLMSEloss, NLLMSEloss_forces
from schnetpack2.custom.metrics import Emetric,Fmetric, uncertainty,uncertainty_forces
import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.representation
from schnetpack2.custom.optimizers import Adam_sdr

def split_sigma_net(m):
  m_list = list(m.children())
  if hasattr(m_list[1], "sigma_net"):
      return (m_list[0],m_list[1].atomref, m_list[1].out_net, m_list[1].sigma_net)
  else:
      return (m_list[0],m_list[1].atomref, m_list[1].out_net)


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

import pdb

class Args:
    def __init__(self):
        self.cuda = True
    
print(torch.cuda.device_count())
args = Args()
args.cuda = True
args.parallel = False
args.batch_size = int(sys.argv[2])*torch.cuda.device_count()
args.property = 'forces'
args.datapath = ''
args.modelpath = os.environ['VSC_SCRATCH']#r'/dev/shm/models'
#args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]


args.features = 128
args.interactions = [2,2]#[2,2,2,3,3,3]
args.cutoff = 20.
args.num_gaussians  = 64
args.start = 0.0
args.model = 'schnet'
args.sc=True
args.sdr = False
args.uncertainty = False
args.uncertainty_forces = False
args.maxz = 100
args.outlayers = 5
args.p = float(sys.argv[5])
args.var_coeff=float(sys.argv[6])
args.order = 2

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
        representation = SchNetMixHop(
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
            sdr = args.sdr,
            start = args.start,
            bn=False,p=args.p, var_coeff=args.var_coeff, debug=False, attention=0, order = args.order)
            
        if not args.uncertainty:
            atomwise_output = spk.custom.atomistic.Energy(args.features, mean=mean, aggregation_mode='sum', stddev=stddev,
                                                    atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers,bn=False,p=args.p, var_coeff=args.var_coeff, uncertainty=args.uncertainty, sdr=args.sdr)
        else:
            atomwise_output = spk.custom.atomistic.MultiOutput(args.features, 2, mean=mean, aggregation_mode='sum', stddev=stddev,
                                                    atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers,bn=False,p=args.p, var_coeff=args.var_coeff, sdr=args.sdr, uncertainty=args.uncertainty)                                           
        print(hasattr(list(atomwise_output.children())[-3][-1].out_net[-1].weight, "dict"))
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    #if parallelize:
    #state = torch.load('./bulkV/64epochs.pth')
    
#    model = nn.DataParallel(model)
    #model.load_state_dict(state,strict=False)
    logging.info(f"The model you built has: {compute_params(model)} parameters")
#    print(hasattr(list(model.children())[1].sigma_net[1].out_net[-1].weight, "dict"))
    return model
    
device = torch.device("cuda")
train_args = args
#spk.utils.set_random_seed(args.seed)
env = SimpleEnvironmentProviderAuto()#ASEEnvironmentProvider(args.cutoff)
#name = r'/dev/shm/data/train' + sys.argv[1]
name = r'/dev/shm/data/bulkVtrain3200'#+ sys.argv[1]
data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
#name = r'/dev/shm/data/test' + sys.argv[1]
name = r'/dev/shm/data/bulkVtest3200'#+ sys.argv[1]
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

model = get_model(train_args, atomref=None, mean=mean, stddev=stddev,
                  train_loader=train_loader,
                  parallelize=args.parallel)
data = DataBunch( train_loader, val_loader,collate_fn=schnetpack2.data.collate_atoms)

#print(hasattr(list(model.children())[-1].sigma_net[1].out_net[-1].weight, "dict"))

learn = Learner(data, model, model_dir=args.modelpath)

learn.split(split_sigma_net)

#learn.purge()
if args.sdr:
    learn.opt_func = Adam_sdr
else:
    learn.opt_func = Adam

mean, stddev = torch.tensor([-1202.6432]), torch.tensor([12.3304])
#mean, stddev = torch.tensor([-1202.6432,0]), torch.tensor([12.3304,1])

if args.uncertainty_forces:
#    learn.loss_func = partial(NLLMSEloss_forces,mean=mean.cuda(), std=stddev.cuda(),kf=100.0,ke=1.0)#best before forces rescaling
    learn.loss_func = partial(NLLMSEloss_forces,mean=mean.cuda(), std=stddev.cuda(),mean_forces=mean_forces.cuda(), std_forces=stddev_forces.cuda(),kf=0.01,ke=10.0)
elif args.uncertainty:
    learn.loss_func = partial(NLLMSEloss,mean=mean.cuda(), std=stddev.cuda(), std_forces=stddev_forces.cuda(),kf=2000.0,ke=20.0)
else:
    learn.loss_func = partial(MSEloss,kf=10.0,ke=0.1)

if args.uncertainty:
    learn.metrics=[partial(Emetric, stddev=stddev.cuda()), partial(Fmetric,stddev=stddev.cuda()), uncertainty,uncertainty_forces]
else:
    learn.metrics=[partial(Emetric, stddev=stddev.cuda()), partial(Fmetric,stddev=stddev.cuda())]
#learn.load('trainbulkVsurf128epochs')
torch.cuda.empty_cache()
print(sys.argv)
 
def plot_recorder(recorder, save_name, skip_start:int=10, skip_end:int=5):
       "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`. Optionally plot and return min gradient"
       import matplotlib.pyplot as plt
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

wd=0
wds = wd#slice(wd)#(0,0,0,0)

lr  = float(sys.argv[4])
lrs = lr
#lrs = (lr,lr,lr,lr)
#learn.lr_find(start_lr=1e-6,end_lr=1e0, num_it=300, wd=wd)
#plot_recorder(learn.recorder, 'uncertainties_wd_{0}'.format(wd))

#print(hasattr(list(learn.model.children())[-3].weight, "dict"))
#try:
#    if args.uncertainty_forces:
#        learn.model.load_state_dict(torch.load(os.path.join(args.modelpath,sys.argv[1]+"_uncertain_forces"+".pth"))['model'])
#    elif args.uncertainty:
#        learn.model.load_state_dict(torch.load(os.path.join(args.modelpath,sys.argv[1]+"_uncertain"+".pth"))['model'])
#    else:
#        learn.model.load_state_dict(torch.load(os.path.join(args.modelpath,sys.argv[1]+".pth"))['model'])
#    print("weights were loaded")
#except:
#    print("no weights were loaded")

#pdb.set_trace()
#with torch.autograd.detect_anomaly():
#learn.fit_one_cycle(cyc_len=int(sys.argv[3]), max_lr=lrs,moms=(0.95, 0.85), div_factor=50.0, pct_start=0.30, wd=wds,no_grad_val=False)

learn.fit_one_cycle(cyc_len=int(sys.argv[3]), max_lr=lrs,moms=(0.95, 0.85), div_factor=150.0, pct_start=0.05, wd=wds,no_grad_val=False)

if args.uncertainty_forces:
    print(learn.save( "bulkVtrain3200"+"_uncertain_forces_"+str(sys.argv[3]), return_path=True))
elif args.uncertainty:
    print(learn.save( "bulkVtrain3200"+"_uncertain_"+str(sys.argv[3])+"_"+str(args.p)+"_"+str(args.var_coeff), return_path=True))
else:
    print(learn.save( "bulkVtrain3200"+"_"+str(sys.argv[3]), return_path=True))