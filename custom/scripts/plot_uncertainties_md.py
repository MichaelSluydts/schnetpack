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
from schnetpack2.custom.loss import MSEloss, logrootloss,MAEloss, NLLMSEloss, NLLMSEloss_forces
from schnetpack2.custom.metrics import Emetric,Fmetric, uncertainty,uncertainty_forces
import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.representation
from schnetpack2.custom.optimizers import Adam_sdr
from schnetpack2.custom.nn.layers import EnsembleModel
from yaff import *
import h5py as h5
from molmod.periodic import periodic
from schnetpack2.md import AtomsConverter
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
from ase import Atoms
import ase.io
import pdb
from schnetpack2.custom.interface.yaff import ML_FF
from schnetpack2.custom.interface.yaff_inherited import SchnetForceField
from tqdm import tqdm
import matplotlib.pyplot as plt

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
args.sdr = True
args.uncertainty = True
args.uncertainty_forces = False
args.maxz = 100
args.outlayers = 5
args.NPT=False

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
            sdr = args.sdr,
            start = args.start,
            bn=False,p=0,debug=False, attention=0)
#        atomwise_output = spk.custom.atomistic.Energy(args.features, mean=mean, aggregation_mode='sum', stddev=stddev,
#                                                atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers,bn=False,p=0.0, uncertainty=args.uncertainty, sdr=args.sdr)
        atomwise_output = spk.custom.atomistic.MultiOutput(args.features, 2, mean=mean, aggregation_mode='sum', stddev=stddev,
                                                atomref=atomref, return_force=True, create_graph=True, train_embeddings=True, n_layers=args.outlayers,bn=False,p=0.0, sdr=args.sdr, uncertainty=args.uncertainty)                                           
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
    
def soft_model_load(model, weight_dict):
    weight_dict_model = model.state_dict()
    
    count=0 
    
    for k,v in weight_dict.items():
        if k in weight_dict_model:
            weight_dict_model[k] = v
        else:
            count+=1
            
    print("number of layers that were not loaded into the model {0}".format(count))
    
    return weight_dict_model
    
def evaluatethings(file, model, env, device,ebins=50,fbins=50):
    atoms = ase.io.read(file,index=':')
    nstructs = len(atoms)

    energies = np.zeros(nstructs)
    results = np.zeros(nstructs)
    sigmas  = np.zeros(nstructs)
    sigmas_ensemble  = np.zeros(nstructs)
        
    forces = []
    forceresults = []
    sigmas_forces = []

    conv = AtomsConverter(environment_provider=env, device=device)

    for i in tqdm(range(nstructs)):
        atin = conv.convert_atoms(atoms=atoms[i])
        result = model(atin, 10)
        nat = atoms[i].get_number_of_atoms()
        forces = np.concatenate([forces,atoms[i].get_forces().flatten()])
        
        forceresults.append(result['dydx'].detach().flatten().cpu().numpy())

        if 'sigma_forces' in result: sigmas_forces.append(result['sigma_forces'].detach().flatten().cpu().numpy())
        
        results[i] = result['y'].detach().cpu().float()/nat
        
        energies[i] = atoms[i].get_total_energy()/nat
        
        if 'sigma' in result: sigmas[i] = result['sigma'].detach().cpu().float()/nat
       
        sigmas_ensemble[i] = result['sigma_ensemble'].detach().cpu().float()/nat
        
    np.save("energies_DFT", results)
    np.save("energies", energies)
    np.save("sigmas", sigmas)
    np.save("sigmas_ensemble", sigmas_ensemble) 
        
    forceresults = np.concatenate(forceresults)  
    if len(sigmas_forces)>0: sigmas_forces = np.concatenate(sigmas_forces)  
    forces = np.array(forces)
    
    np.save("forces_DFT", forces)
    np.save("forces", forceresults)
    np.save("sigmas_forces", sigmas_forces)   
    
#    fdiff = (forces-forceresults)
#    diff = (results-energies)
#    relative = diff/energies
#    frelative = fdiff*2./(np.abs(forces)+np.abs(forceresults)) # 2x-y/x + y
#    fig, ax = plt.subplots(2,3,figsize=(16,8))
#    ax[0,0].hist(energies,bins=ebins, color='blue',label='DFT')
#    ax[0,0].hist(results,bins=ebins, color='orange',alpha=0.7,label='SchNet')
#    ax[0,0].legend()
#    #ax[0,0].set_xlim(np.min([np.min(results),np.min(energies)]),np.max([np.max(results),np.max(energies)]))
#    ax[0,0].set_xlabel('Energies (eV/atom)',weight='bold',fontsize=14)
#    ax[0,1].hist(diff,bins=ebins)
#    ax[0,1].set_xlabel('Error (eV/atom)',weight='bold',fontsize=14)
#    ax[0,2].hist(relative,bins=ebins)
#    ax[0,2].set_xlabel('Relative error (eV/atom)',weight='bold',fontsize=14)
#    ax[1,0].hist(forces,bins=fbins,label='DFT')
#    ax[1,0].hist(forceresults,bins=fbins, color='orange',alpha=0.7,label='SchNet')
#    ax[1,0].legend()
#    ax[1,0].set_xlabel('Forces (eV/$\AA$)',weight='bold',fontsize=14)
#    ax[1,1].hist(fdiff,bins=fbins)
#    ax[1,1].set_xlabel('Error (eV/$\AA$)',weight='bold',fontsize=14)
#    ax[1,2].hist(frelative,bins=fbins)
#    ax[1,2].set_xlabel('Relative error (eV/$\AA$)',weight='bold',fontsize=14)
#    fig.suptitle('Error report for ' + file,weight='bold',fontsize=18)
#    plt.show()
    
    
device = torch.device("cuda")
train_args = args
spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)
#name = r'/dev/shm/data/train' + sys.argv[1]
name = r'/dev/shm/data/bulkVtrain3200'
data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
#name = r'/dev/shm/data/test' + sys.argv[1]
name = r'/dev/shm/data/bulkVtest3200'
data_val = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])

#vasprun_dir = r"/data/gent/vo/000/gvo00003/shared/michiel/Gepure/I/1.0/vasprun.xml"
#vasprun_dir = r"/data/gent/vo/000/gvo00003/shared/michiel/Gepure/bulk/1000/1.0/vasprun.xml"
vasprun_dir = r"/data/gent/vo/000/gvo00003/shared/michiel/interstitials/I/V0/64/10000/1.0/vasprun.xml"
#vasprun_dir = r"/data/gent/vo/000/gvo00003/shared/michiel/Gepure/V13/2000-216/200NPT/vasprun.xml"

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
mean, stddev = torch.tensor([-1202.6432,0]), torch.tensor([12.3304,1])
#mean, stddev = torch.tensor([-1202.6432]), torch.tensor([12.3304])

print(mean,stddev)
print(mean_forces,stddev_forces)

model = get_model(train_args, atomref=None, mean=mean, stddev=stddev,
                  train_loader=train_loader,
                  parallelize=args.parallel)
data = DataBunch( train_loader, val_loader,collate_fn=schnetpack2.data.collate_atoms)

#pdb.set_trace()
#print(hasattr(list(model.children())[-1].sigma_net[1].out_net[-1].weight, "dict"))

learn = Learner(data, model, model_dir=args.modelpath)

learn.split(split_sigma_net)

#learn.purge()
if args.sdr:
    learn.opt_func = Adam_sdr
else:
    learn.opt_func = Adam

mean, stddev = torch.tensor([-1202.6432]), torch.tensor([12.3304])

if args.uncertainty_forces:
#    learn.loss_func = partial(NLLMSEloss_forces,mean=mean.cuda(), std=stddev.cuda(),kf=100.0,ke=1.0)#best before forces rescaling
    learn.loss_func = partial(NLLMSEloss_forces,mean=mean.cuda(), std=stddev.cuda(), std_forces=stddev_forces.cuda(),kf=0.01,ke=10.0)
elif args.uncertainty:
    learn.loss_func = partial(NLLMSEloss,mean=mean.cuda(), std=stddev.cuda(), std_forces=stddev_forces.cuda(),kf=2000.0,ke=20.0)
else:
    learn.loss_func = partial(MSEloss,kf=1.0,ke=0.1)

learn.metrics=[partial(Emetric, stddev=stddev.cuda()), partial(Fmetric,stddev=stddev.cuda()), uncertainty,uncertainty_forces]

#learn.lr_find(start_lr=1e-6,end_lr=1e0, num_it=300, wd=wd)
#plot_recorder(learn.recorder, 'uncertainties_wd_{0}'.format(wd))

#print(hasattr(list(learn.model.children())[-3].weight, "dict"))
try:
    if args.uncertainty_forces:
        learn.model.load_state_dict(torch.load(os.path.join(args.modelpath,sys.argv[1]+"_uncertain_forces"+".pth"))['model'])
        #learn.model.load_state_dict(soft_model_load(learn.model, torch.load(os.path.join(args.modelpath,sys.argv[1]+"_uncertain_forces"+".pth"))['model']))
    elif args.uncertainty:
        #learn.model.load_state_dict(torch.load(os.path.join(args.modelpath,sys.argv[1]+"_uncertain"+".pth"))['model'])
        learn.model.load_state_dict(soft_model_load(learn.model, torch.load(os.path.join(args.modelpath,sys.argv[1]+"_uncertain_128_0.1_0.0"+".pth"))['model']))
        print("loaded model "+os.path.join(args.modelpath,sys.argv[1]+"_uncertain_128_0.1_0.0"+".pth"))
    else:
        learn.model.load_state_dict(torch.load(os.path.join(args.modelpath,sys.argv[1]+".pth"))['model'])
    print("weights were loaded")
except Exception as e:
   print( "Error: %s" % str(e) )

ensemble_model = EnsembleModel(learn.model ,mean, stddev, mean_forces, stddev_forces).cuda()

env = ASEEnvironmentProvider(args.cutoff)
conv = schnetpack2.md.AtomsConverter(device=device)

#poscarpath = '/dev/shm/data/vasprun215.xml'

#atom = ase.io.read(poscarpath,index=0)

#MD run
#system = atoms2yaff(atom)
#steps = 500

#mlff = ML_FF(system,model)
#mlff = ML_FF(atom, model, conv, env)
#ensemble_model.train()
#ensemble_model.model.output_modules.training=False
ensemble_model.train()
#mlff = SchnetForceField('schnetforcefield', atom, ensemble_model, conv, env)

evaluatethings(vasprun_dir, ensemble_model, env, device,ebins=50,fbins=50)
#model.eval()
#mlff = SchnetForceField('schnetforcefield', atom, model, conv, env)

#if args.NPT:
#  npt = mlff.NPT(steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')
#else:
#  nvt = mlff.NVT(steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')
