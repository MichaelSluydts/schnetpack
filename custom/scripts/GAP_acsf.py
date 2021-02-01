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

def par_func(ind1, ind2, vector1, vector2):
    mat1, mat2 = vectors1[ind1], vectors2[ind2]
    distance_matrix = cdist_torch(mat1, mat2)
    row_ind, col_ind = linear_sum_assignment(distance_matrix.detach().cpu().numpy())
    similarities[ind, ind2] = torch.exp(-distance_matrix[row_ind, col_ind]).sum()
    
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
            #pdb.set_trace()
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

def cdist_torch(vectors, vectors2):
    """ computes the coupled distances
    Args:
        vectors  is a N x D matrix
        vectors2 is a M x D matrix
        
        returns a N x M matrix
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors2)) + vectors.pow(2).sum(dim=1).view(-1, 1) + vectors2.pow(2).sum(dim=1).view(1, -1)
    return distance_matrix.abs()
    
def logdet_torch(x): 
    x_chol = torch.cholesky(torch.einsum('bik,bjk->bij',x,x))
    det = torch.diagonal(x_chol, dim1=-2, dim2=-1).log().sum(-1)
    return det

class AtomExactMarginalLogLikelihood(nn.Module):
    def __init__(self, likelihood, model, kf=0.1):
        super(AtomExactMarginalLogLikelihood, self).__init__()
        self.mll = likelihood#gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        self.kf  = kf
        self.model = model
        
    def forward(self, results, train_y):
#        pdb.set_trace()
        if "dydx" in results:
            return self.mll(train_y["energy"]).mean() + self.kf*(results["dydx"]-train_y["forces"]).pow(2).mean()
            #return (output[0] - train_y_nrm).pow(2).mean() + self.kf*(forces-train_y["forces"]).pow(2).mean()
        else:
            return self.mll().mean()
     
class LikelihoodGP(nn.Module):
    def __init__(self, model):
        super(LikelihoodGP, self).__init__()
        self.model = model
    
    def forward(self, targets):
        #K      = self.model.RBF_kernel(x,x) + torch.eye(len(x))*self.model.sigma
#        pdb.set_trace()
#        K_inv  = torch.inverse(covar_mat)
#        targets = (targets-self.model.means)/(self.model.stddevs+1e-5)
#        pdb.set_trace()
        #return 1/2*(torch.einsum("in,ij,jm->nm", targets-mean, self.model.K_inv, targets-mean) - logdet_torch(self.model.K_inv[None,...]) + mean.shape[-1]*math.log(2*math.pi))
        return 1/2*(self.model.targets.dot(self.model.alpha) + 2*torch.diagonal(self.model.L).sum() + 1*math.log(2*math.pi))
   
class RegressionGPModel(nn.Module):
    def __init__(self, n_in):
        super(RegressionGPModel, self).__init__()
#        self.kernel = torch.eye(10)
#        self.mean = torch.zeros(n_in)
#        self.targets = torch.ones(10)
        data_points = torch.eye(10, n_in)
        gamma   = torch.ones(1, n_in)*(-2)
        sigma   = torch.ones(1)*(-3)
        targets = torch.ones(n_in).normal_()
        L     = torch.ones(10,10)
        alpha = torch.ones(10) 
        self.n_in  = n_in
        
        means   = torch.ones(n_in,1)
        stddevs = torch.ones(n_in,1)
        
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
        
class CombinedModel(nn.Module):
    def __init__(self, reps, model, include_forces = True):
        super(CombinedModel, self).__init__()
        self.reps  = reps
        self.model = model
        self.include_forces = include_forces
        
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
        
        if self.include_forces:
            forces = -grad(results["y"], x["_positions"], grad_outputs=torch.ones_like(output[0]), create_graph=True, retain_graph=True)[0]
      
            results["dydx"]  = forces 
      
        return results

def read_data_Tom(dirname):
    atms = []
    for transition in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, transition)):
            for run in os.listdir(os.path.join(dirname, transition)):
                if os.path.isdir(os.path.join(dirname, transition, run)):
                    filename = os.path.join(dirname, transition, run, "vasprun.xml")
                    print(filename)
                    atms.extend(ase.io.read(filename, format="vasp-xml",index=':'))
              
    with connect(os.path.join(dirname, 'perovskites.db')) as db:
        for atm in atms: db.write(atm)
        
def plot_all_reps(dataset, reps):
    import numpy as np
    
    data_loader = schnetpack2.custom.data.AtomsLoader(dataset, batch_size=len(dataset),
                                    num_workers=9*torch.cuda.device_count(), pin_memory=True)
    
    data_iter = iter(data_loader)
                           
    for x in data_iter:
        representations = reps(x)
    
    np.save("representations_acsf.npy", representations.detach().cpu().numpy())
    
def generate_split_file(split_file):
    import numpy as np
    ids_train = np.array([1,2,3,5,6,7,8])
    ids_val   = np.array([0,4,9])
    
    train_idx = np.hstack([ids_train, ids_train+10, ids_train+20]).tolist()
    val_idx   = np.hstack([ids_val  , ids_val+10  , ids_val+20]).tolist()
    test_idx  = []
    
    np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        
spk.utils.set_random_seed(666)

cutoff = 7.0
data_dir = r"/dev/shm/data"#os.environ["VSC_SCRATCH"]
batch_size = 8
training_iter = 5

print(os.path.join(data_dir, "perovskites.db"))
data = Perovskites(data_dir, cutoff, properties = ["energy", "forces"], subset=None, collect_triples = True)

# split in train and val
num_val = len(data)//5+1
num_train = len(data) - num_val
split_file = "split.npz"
generate_split_file(split_file)
data_train, data_val, _ = data.create_splits(num_train, num_val, split_file=split_file)

train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=num_train, num_workers=9*torch.cuda.device_count(), pin_memory=True)
val_loader   = schnetpack2.custom.data.AtomsLoader(data_val  , batch_size=num_val  , num_workers=9*torch.cuda.device_count(), pin_memory=True)

## create model
reps = rep.BehlerSFBlock(elements={i+1 for i in range(90)}, n_radial=5, n_angular=5, zetas={1}, cutoff_radius=cutoff)

D = reps.n_symfuncs
print(D)

#plot_all_reps(data, reps)

model = RegressionGPModel(D)

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

model.train()
likelihood = LikelihoodGP(model)

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': trainable_params},
], lr=0.3)

combined_model = CombinedModel(reps, model).cuda()

loss_func = AtomExactMarginalLogLikelihood(likelihood, model, kf=10000.0)

data_iter = iter(train_loader)

for x in data_iter:
#    pdb.set_trace()
    print("energy mean: {}, std: {}".format(x["energy"].mean(),x["energy"].std())) 
    print("forces mean: {}, std: {}".format(x["forces"].mean((0,1)),x["forces"].std((0,1)))) 

for i in range(training_iter):
    data_iter = iter(train_loader)
    
    for x in data_iter:
        optimizer.zero_grad()
        x["energy"] = x["energy"].squeeze(-1)
        x["forces"] = x["forces"].squeeze(-1)
        results = combined_model(x,x["energy"])
        loss = loss_func(results, x)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            (1/model.gamma.exp()).mean(),
            model.sigma.exp()
        ))
        optimizer.step()

combined_model.eval()
likelihood.eval()
    
data_iter = iter(train_loader)

#pdb.set_trace()

for x in data_iter:
    optimizer.zero_grad()
    x["energy"] = x["energy"].squeeze(-1)
    x["forces"] = x["forces"].squeeze(-1)
    results = combined_model(x, None)
    loss = loss_func(results, x)
    optimizer.zero_grad()
    print('mae energy: %.3f   mae forces: %.3f' % (
        (x["energy"] - results["y"]).abs().mean().item(),
        (x["forces"] - results["dydx"]).abs().mean().item()
    ))
    
data_iter = iter(val_loader)

#pdb.set_trace()

for x in data_iter:
    optimizer.zero_grad()
    x["energy"] = x["energy"].squeeze(-1)
    x["forces"] = x["forces"].squeeze(-1)
    results = combined_model(x, None)
    loss = loss_func(results, x)
    optimizer.zero_grad()
    print('mae energy: %.3f   mae forces: %.3f' % (
        (x["energy"] - results["y"]).abs().mean().item(),
        (x["forces"] - results["dydx"]).abs().mean().item()
    ))
    
torch.save(combined_model.state_dict(), os.path.join(data_dir, "GAP_model"))