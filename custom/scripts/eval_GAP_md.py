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
from torch.nn.parameter import Parameter
from torch.autograd import grad, Variable

import schnetpack as spk
from torch.optim import Adam, SGD
from torch.utils.data.sampler import RandomSampler

from schnetpack2.environment import ASEEnvironmentProvider
from schnetpack2.datasets import MaterialsProject
from schnetpack2.utils import to_json, read_from_json, compute_params
import schnetpack2.custom.representation as rep

from schnetpack2.custom.representation.schnet import SchNetInteraction
from schnetpack2.custom.loss import MSEloss, logrootloss,MAEloss, NLLMSEloss, NLLMSEloss_forces
from schnetpack2.custom.metrics import Emetric,Fmetric, uncertainty,uncertainty_forces
import schnetpack2.custom.data
from schnetpack2.custom.datasets.perovskites import Perovskites
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.representation
from schnetpack2.custom.optimizers import Adam_sdr
from schnetpack2.custom.nn.layers import EnsembleModel
from schnetpack2.custom.interface.yaff import ML_FF
from schnetpack2.custom.interface.yaff_inherited import SchnetForceField

from yaff import *
import h5py as h5
from molmod.periodic import periodic
from schnetpack2.custom.md import AtomsConverter
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
from ase import Atoms
import ase.io
import pdb

from scipy.optimize import linear_sum_assignment

def cdist_torch(vectors, vectors2):
    """ computes the coupled distances
    Args:
        vectors  is a N x D matrix
        vectors2 is a M x D matrix
        
        returns a N x M matrix
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors2)) + vectors.pow(2).sum(dim=1).view(-1, 1) + vectors2.pow(2).sum(dim=1).view(1, -1)
    return distance_matrix.abs()

def material_distances(vectors1, vectors2):
    """ computes the coupled distances
    Args:
        vectors  is a B x N x d matrix
        vectors2 is a D x M x d matrix
        
        returns a B x D matrix
    """
#    from multiprocessing import Pool
#    from itertools import product
#    from functools import partial
    
    B,N,d = vectors1.shape
    D,M,d = vectors2.shape
    
    similarities = Variable(torch.zeros(B,D))
    
#    hungarian = Hungarian()
    
    for ind, mat1 in enumerate(vectors1):
        for ind2, mat2 in enumerate(vectors2):
            distance_matrix = cdist_torch(mat1, mat2)
            row_ind, col_ind = linear_sum_assignment(distance_matrix.detach().cpu().numpy())
            #indices = torch.tensor([list(i) for i in hungarian.get_results()])
            #similarities[ind, ind2] = torch.exp(-distance_matrix[indices[:,0], indices[:,1]]).sum()
            similarities[ind, ind2] = torch.exp(-distance_matrix[row_ind, col_ind]).sum()
            #distance_matrix[]
            
#    pool = Pool() #defaults to number of available CPU's
#    chunksize = 20 #this may take some guessing ... take a look at the docs to decide
#    for ind, res in enumerate(pool.imap(partial(par_func, vector1=vectors1, vector2=vectors2) , product(range(B), range(D))), chunksize):
#        similarities[ind] = res
#        
#    pool.close()
#    pool.join()   
                
    return similarities/max(M,N)

class RegressionGPModel(nn.Module):
    def __init__(self, n_in, n_points, A):
        super(RegressionGPModel, self).__init__()
#        self.kernel = torch.eye(10)
#        self.mean = torch.zeros(n_in)
#        self.targets = torch.ones(10)
        data_points = torch.ones(n_points, A, n_in)
        gamma   = torch.ones(1, n_in)*(-2)
        sigma   = torch.ones(1)*(-3)
        targets = torch.ones(n_points).normal_()
        L     = torch.ones(n_points,n_points)
        alpha = torch.ones(n_points) 
        self.n_in  = n_in
        
        means   = torch.tensor(0.0)
        stddevs = torch.tensor(1.0)
        
        self.register_parameter("gamma", Parameter(gamma))
        self.register_parameter("sigma", Parameter(sigma))
        
        self.register_buffer("means", means)
        self.register_buffer("stddevs", stddevs)
        
        self.register_buffer("data_points", data_points)
        self.register_buffer("targets", targets)
        self.register_buffer("alpha", alpha)
        self.register_buffer("L", L)
                
    def forward(self,x, targets=None):
#        pdb.set_trace()
        if self.training and targets is not None:
            self.data_points = x.detach().data
            self.targets = targets.detach().data
        
            K          = self.Material_kernel(self.data_points,self.data_points) + torch.eye(len(self.data_points))*(self.sigma.exp()+1e-3)
            self.L     = torch.cholesky(K, upper=False)
#            pdb.set_trace()
            self.alpha = torch.einsum("in,ij,j->n", torch.inverse(self.L), torch.inverse(self.L), self.targets)
            #self.K_inv  = torch.inverse(K)

        K_xnew = self.Material_kernel(self.data_points, x)

        v = torch.inverse(self.L).mm(K_xnew)
        mean   = torch.einsum("in, i->n", K_xnew, self.alpha)#torch.einsum("ni,ij,j->n", K_xnew, self.K_inv, self.targets.squeeze(-1))
        var = torch.diagonal(self.Material_kernel(x, x)) - torch.einsum("in,in->n",v,v)#torch.einsum("ni,ij,nj->n", K_xnew, self.K_inv, K_xnew)
#        pdb.set_trace()
        return mean, var
            
    def RBF_kernel(self, x1, x2):
        return cdist_torch(x1*self.gamma, x2*self.gamma)
        
    def Material_kernel(self, x1, x2):
        return material_distances(x1*self.gamma.exp(), x2*self.gamma.exp())
        
    def likelihood(self):
        K      = self.RBF_kernel(self.data_points,self.data_points) + torch.eye(len(self.data_points))*self.sigma
        K_inv  = torch.inverse(K)        
        return -1/2*(torch.einsum("ni,ij,mj->nm", self.targets, K_inv, self.targets) + logdet_torch(K) + self.n_in*math.log(2*math.pi))
        
class BiasPotential(nn.Module):
    def __init__(self, omega, tau, sigma):
        self.omega = omega
        self.tau   = tau
        self.sigma = sigma

        self.tau_last = 0
        self.time = 0
        self.means = []
        
    def forward(self, x):
    
        if self.time//self.tau>self.tau_last:
            self.tau_last+=1
            self.means.append(x)
        
        self.time += 1
        
        return self.tau*self.omega*torch.exp(-1/2*((torch.cat(means)-x)/self.sigma).pow(2)).sum()
        
            
class CombinedModel(nn.Module):
    def __init__(self, reps, model, include_forces = True, MTD = False, rep_final=None, tau=1, omega=1, sigma=1):
        super(CombinedModel, self).__init__()
        self.reps  = reps
        self.model = model
        self.include_forces = include_forces
        
        self.MTD = MTD
        
        if self.MTD:
            self.rep_final = rep_final
            self.bias_potential = Bias_Potential(omega, tau, sigma)
        
    def forward(self, x, targets=None):
        results = {}
    
        x["_positions"].requires_grad_()
      
        x_rep = self.reps(x) 
        B,A,D = x_rep.shape
        x_rep = x_rep.view(B*A,D)
        
        mean, std = x_rep.mean(0), x_rep.std(0)
        x_rep = (x_rep-mean)/(std+1e-5)
        
        x_rep = x_rep.view(B,A,D)
        
        if targets is not None:
            mean_y, std_y = targets.mean(0), targets.std(0)
            targets = (targets-mean_y)/(std_y+1e-5)
        
            self.model.means   = mean_y
            self.model.stddevs = std_y
        
        output  = model(x_rep, targets)
        
        results["y"]     = output[0]*self.model.stddevs + self.model.means
        results["sigma"] = (output[1]+1e-5).sqrt()*self.model.stddevs
        
        if self.MTD:
            CV = self.model.Material_kernel(x_rep[None,...], self.rep_final[None,...])
            
            energy = results["y"] + self.bias_potential(CV)
        else:
            energy = results["y"]
                
        
        if self.include_forces:
            forces = -grad(energy, x["_positions"], grad_outputs=torch.ones_like(output[0]), create_graph=True, retain_graph=True)[0]
      
            results["dydx"]  = forces 
      
        return results
        
def rep_from_file(rep, env, filename):
    at     = ase.io.read(filename)
    dct    = schnetpack2.custom.data.prepare_atom(at, env)
    inputs = schnetpack2.custom.data.collate_atoms([dct])
    return rep(inputs).detach()
    

import pdb

class Args:
    def __init__(self):
        self.cuda = True
        
pdb.set_trace()
    
print(torch.cuda.device_count())
args = Args()
args.cuda = True
args.parallel = False
args.batch_size = int(sys.argv[1])*torch.cuda.device_count()
args.property = 'forces'
args.datapath = '/dev/shm/data'
args.modelpath = os.environ['VSC_SCRATCH']
args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]

args.features = 32
args.interactions = 5#[2,2,2,3,3,3]
args.cutoff = 7.
args.num_gaussians  = 32
args.start = 0.0
args.model = 'schnet'
args.sc=True
args.maxz = 100
args.outlayers = 5
args.num_inducing_points = 30
args.sdr = False
args.uncertainty = False
args.uncertainty_forces = False
args.maxz = 100
args.outlayers = 5
args.NPT = False
#args.p = float(sys.argv[5])
#args.var_coeff=float(sys.argv[6])

device = "cpu"#torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
print(device)
if "cuda" in device: torch.cuda.set_device(device)
    
#device = torch.device("cuda")
train_args = args
#spk.utils.set_random_seed(args.seed)
env = ASEEnvironmentProvider(args.cutoff)

if not os.path.isdir(args.modelpath):
    os.makedirs(args.modelpath)
    
if args.split_path is not None:
    copyfile(args.split_path, split_path)
    
data = Perovskites(args.datapath, args.cutoff, properties = ["energy", "forces"], subset=None, collect_triples = True)

# split in train and val
num_val = len(data)//5+1
num_train = len(data) - num_val
split_file = "split.npz"
#generate_split_file(split_file)
data_train, data_val, _ = data.create_splits(num_train, num_val, split_file=split_file)

train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=num_train, num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader   = schnetpack2.custom.data.AtomsLoader(data_val  , batch_size=num_val  , num_workers=9*torch.cuda.device_count(), pin_memory=True)

#mean, stddev = train_loader.get_statistics('energy', False)
#mean_forces, stddev_forces = train_loader.get_statistics('forces', True)

mean, stddev = torch.tensor([-1202.6432]), torch.tensor([12.3304])
mean_forces,stddev_forces = torch.tensor([7.8785e-06, 8.8354e-07, 3.8550e-07]), torch.tensor([0.6644, 0.6627, 0.6716])

print(mean,stddev)
print(mean_forces,stddev_forces)

reps = rep.BehlerSFBlock(elements={i+1 for i in range(90)}, n_radial=5, n_angular=5, zetas={1}, cutoff_radius=args.cutoff)

D = reps.n_symfuncs
print(D)
A = 40

rep_final= rep_from_file(rep, env, '/data/gent/gvo000/gvo00003/shared/InputMLpot/AtoE/09/vasprun.xml')

model = RegressionGPModel(D, len(data_train),A)

combined_model = CombinedModel(reps, model, MTD = False, rep_final=rep_final).cpu()

pdb.set_trace()

try:
    combined_model.load_state_dict(torch.load(os.path.join(args.datapath, "GAP_model")))
    print("weights were loaded")
except:
    print("no weights were loaded")

conv = schnetpack2.custom.md.AtomsConverter(device=device, environment_provider=env, collect_triples=True)

poscarpath = '/data/gent/gvo000/gvo00003/shared/InputMLpot/AtoE/00/vasprun.xml'

atom = ase.io.read(poscarpath,index=0)

steps = 500

combined_model.eval()
mlff = SchnetForceField('GAPforcefield', atom, combined_model, conv, env)

if args.NPT:
  npt = mlff.NPT(steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')
else:
  nvt = mlff.NVT(steps, nprint = 10, dt = 1, temp = 1200, start = 0, name = 'run')
