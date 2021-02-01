#import schnetpack2.atomistic.output_modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler
from torch.nn.parameter import Parameter

import schnetpack as spk
import schnetpack2.custom.representation as rep
from schnetpack2.environment import ASEEnvironmentProvider

from schnetpack2.custom.datasets.perovskites import Perovskites
import schnetpack2.custom.data
from schnetpack2.custom.GP.AtomKernel import RBFKernelMaterial

import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy

#import pyro.contrib.gp as gp
#from gpytorch import settings
from scipy.optimize import linear_sum_assignment

import os
import ase.io
from ase.db import connect
import copy

import math

import pdb

class AtomExactMarginalLogLikelihood(nn.Module):
    def __init__(self, likelihood, model, kf=0.1):
        super(AtomExactMarginalLogLikelihood, self).__init__()
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.kf  = kf
        self.model = model
        
    def forward(self, output, forces, train_y):
        #pdb.set_trace()
        if forces is not None:
            return -self.mll(output, train_y["energy"]).mean() + self.kf*(forces-train_y["forces"]).pow(2).mean()
        else:
            return -self.mll(output, train_y["energy"]).mean()
            
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)     
        
class CombinedModel(nn.Module):
    def __init__(self, reps, model):
        super(CombinedModel, self).__init__()
        self.reps  = reps
        self.model = model
        
    def forward(self, x, targets=None):
        atom_masks = x["_atom_mask"]
        x["_positions"].requires_grad_()
      
        x_rep = self.reps(x) 
        B,A,D = x_rep.shape
        x_rep = x_rep.view(B*A,D)
        
        mean, std = x_rep.mean(0), x_rep.std(0)
        x_rep = (x_rep-mean)/(std+1e-5)
        
#        x_rep = x_rep.view(B,A,D)
        
        if targets is not None:
            mean_y, std_y = targets.mean(0), targets.std(0)
            targets = (targets-mean_y)/(std_y+1e-5)
        
            self.means   = mean_y
            self.stddevs = std_y
        
        pdb.set_trace()
        
        output  = model(x_rep)

        mean = (output.mean.view(B, A) * atom_masks).sum(1)
        covar_mat = (output.stddev.pow(2).view(B, A) * atom_masks).sum(1)[...,None] * torch.eye(B)
        output_tot = gpytorch.distributions.MultivariateNormal(mean*self.stddevs + self.means, covar_mat*self.stddevs)        
      
        #output  = model(x_rep.sum(1))
        
        return output_tot

def read_data_Tom(dirname):
    atms = []
    for transition in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, transition)):
            for run in os.listdir(os.path.join(dirname, transition)):
                if os.path.isdir(os.path.join(dirname, transition, run)):
                    filename = os.path.join(dirname, transition, run, "vasprun.xml")
                    print(filename)
                    atms.append(ase.io.read(filename, format="vasp-xml"))
              
    with connect(os.path.join(dirname, 'perovskites.db')) as db:
        for atm in atms: db.write(atm)
        
#dirname = r"/data/gent/vo/000/gvo00003/shared/InputMLpot/"       
#read_data_Tom(dirname)

# load qm9 dataset and download if necessary
cutoff = 7.0
data_dir = r"/dev/shm/data"#os.environ["VSC_SCRATCH"]
batch_size = 8
training_iter = 1

print(os.path.join(data_dir, "perovskites.db"))
data = Perovskites(data_dir, cutoff, properties = ["energy", "forces"], subset=None, collect_triples = True)

# split in train and val
num_val = len(data)//5+1
num_train = len(data) - num_val
data_train, data_val, _ = data.create_splits(num_train, num_val)
train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=num_train, #sampler=data_train,
                                    num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=num_val, num_workers=9*torch.cuda.device_count(), pin_memory=True)

## create model
reps = rep.BehlerSFBlock(elements={i+1 for i in range(90)}, n_radial=12, n_angular=10, zetas={1})

D = reps.n_symfuncs
print(D)

data_iter = iter(train_loader)

for x in data_iter:
    print("energy mean: {}, std: {}".format(x["energy"].mean(),x["energy"].std())) 
    print("forces mean: {}, std: {}".format(x["forces"].mean((0,1)),x["forces"].std((0,1)))) 
    x_rep = reps(x)
    
B,A,D = x_rep.shape
x_rep = x_rep.view(B*A,D)

mean, std = x_rep.mean(0), x_rep.std(0)
x_rep = (x_rep-mean)/(std+1e-5)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_rep.detach().data, x["energy"].squeeze(-1).detach().data, likelihood)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
#    {'params': likelihood.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
trainable_params = filter(lambda p: p.requires_grad, model.parameters())

combined_model = CombinedModel(reps, model)

combined_model.train()
likelihood.train()

loss_func = AtomExactMarginalLogLikelihood(likelihood, model, kf=0.0)

data_iter = iter(train_loader)
with gpytorch.settings.debug(False):
    for i in range(training_iter):
        data_iter = iter(train_loader)
        
        for x in data_iter:
            optimizer.zero_grad()
            x["energy"] = x["energy"].squeeze(-1)
            x["forces"] = x["forces"].squeeze(-1)
            output = combined_model(x,x["energy"] )
            #forces = -grad(output.mean, x["_positions"], grad_outputs=torch.ones_like(output.mean), create_graph=True, retain_graph=True)[0]
            loss = loss_func(output, None, x)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()

combined_model.eval()
likelihood.eval()
    
data_iter = iter(train_loader)

for x in data_iter:
    optimizer.zero_grad()
    x["energy"] = x["energy"].squeeze(-1)
    x["forces"] = x["forces"].squeeze(-1)
    output = combined_model(x, None)
#    forces = -grad(output.mean, x["_positions"], grad_outputs=torch.ones_like(output.mean), create_graph=False, retain_graph=False)[0]
    loss = loss_func(output, None, x)
    optimizer.zero_grad()
    pdb.set_trace()
    print('Loss: %.3f   mae energy: %.3f   mae forces: %.3f' % (
        loss.item(), (x["energy"] - output[0]).abs().mean().item(),
    ))
    
data_iter = iter(val_loader)

for x in data_iter:
    optimizer.zero_grad()
    x["energy"] = x["energy"].squeeze(-1)
    x["forces"] = x["forces"].squeeze(-1)
    output = combined_model(x, None)
#    forces = -grad(output.mean, x["_positions"], grad_outputs=torch.ones_like(output.mean), create_graph=False, retain_graph=False)[0]
    loss = loss_func(output, None, x)
    optimizer.zero_grad()
    pdb.set_trace()
    print('Loss: %.3f   mae energy: %.3f   mae forces: %.3f' % (
        loss.item(), (x["energy"] - output[0]).abs().mean().item(),
    ))