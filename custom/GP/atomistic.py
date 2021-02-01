r"""
Classes for output modules.
"""

from collections import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

import schnetpack2.nn.activations
import schnetpack2.nn.base
import schnetpack2.nn.blocks
from schnetpack2.data import Structure

import schnetpack2.custom.nn.base
import schnetpack2.custom.nn.activations
from schnetpack2.custom.nn.cutoff import PhysNetCutoff

import gpytorch.distributions

import pdb

k_e = 0.62415089925#0.02293710445# #check units once more

class AtomisticModel(nn.Module):
    """
    Assembles an atomistic model from a representation module and one or more output modules.

    Returns either the predicted property or a dict (output_module -> property),
    depending on whether output_modules is a Module or an iterable.

    Args:
        representation(nn.Module): neural network that builds atom-wise features of shape batch x atoms x features
        output_modules(nn.OutputModule or iterable of nn.OutputModule): predicts desired property from representation


    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()

        self.representation = representation

        if isinstance(output_modules, Iterable):
            self.output_modules = nn.ModuleList(output_modules)
            self.requires_dr = False
            for o in self.output_modules:
                if o.requires_dr:
                    self.requires_dr = True
                    break
        else:
            self.output_modules = output_modules
            self.requires_dr = output_modules.requires_dr

    def forward(self, inputs):
        r"""
        predicts property

        """
        if self.requires_dr:
            inputs[Structure.R].requires_grad_()
        t = self.representation(inputs)

        if type(t) != tuple:
            t = t, None

        inputs['representation'], inputs['dist_vec'] = t


        if isinstance(self.output_modules, nn.ModuleList):
            outs = {}
            for output_module in self.output_modules:
                outs[output_module] = output_module(inputs)
        else:
            outs = self.output_modules(inputs)

        return outs


class OutputModule(nn.Module):
    r"""
    Base class for output modules.

    Args:
        n_in (int): input dimension
        n_out (int): output dimension
        requires_dr (bool): specifies if the derivative of the ouput is required
    """

    def __init__(self, n_in, n_out, requires_dr=False):
        self.n_in = n_in
        self.n_out = n_out
        self.requires_dr = requires_dr
        super(OutputModule, self).__init__()

    def forward(self, inputs):
        r"""
        Should be overwritten
        """
        raise NotImplementedError


class Atomwise(OutputModule):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    Args:
        n_in (int): input dimension of representation (default: 128)
        n_out (int): output dimension of target property (default: 1)
        pool_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of nn in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output network.
                                          If `None`, divide neurons by 2 in each layer. (default: None)
        activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
        return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
        requires_dr (bool): True, if derivative w.r.t. atom positions is required (default: False)
        mean (torch.FloatTensor): mean of property (default: None)
        stddev (torch.FloatTensor): standard deviation of property (default: None)
        atomref (torch.Tensor): reference single-atom properties
        max_z (int): only relevant only if train_embeddings is true.
                     Specifies maximal nuclear charge of atoms. (default: 100)
        outnet (callable): Network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                           not normalized. If set to None (default), a pyramidal network is generated automatically.
        train_embeddings (bool): if set to true, atomref will be ignored and learned from data (default: None)

    Returns:
        tuple: prediction for property

        If return_contributions is true additionally returns atom-wise contributions.

        If requires_dr is true additionally returns derivative w.r.t. atom positions.

    """

    def __init__(self, n_in=128, n_out=1, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=schnetpack2.nn.activations.shifted_softplus, return_contributions=False,
                 requires_dr=False, create_graph=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None,
                 train_embeddings=False, bn=False, p=0, uncertainty=False, sdr=False, var_coeff=0.1):
        super(Atomwise, self).__init__(n_in, n_out, requires_dr)

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.return_contributions = return_contributions

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(torch.from_numpy(atomref.astype(np.float32)),
                                                        freeze=train_embeddings)
        elif train_embeddings:
            temp = np.zeros((max_z, 1), dtype=np.float32)
            temp[32] = -19.0329202806
            self.atomref = nn.Embedding.from_pretrained(torch.from_numpy(temp),
                                                        freeze=train_embeddings)
        else:
            self.atomref = None


        if outnet is None:
            self.out_net = nn.Sequential(
                schnetpack2.nn.base.GetItem('representation'),
                schnetpack2.custom.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation, bn=bn, p=p, sdr=sdr, var_coeff=var_coeff)
            )        
        else:
            self.out_net = outnet
        
        self.uncertainty = uncertainty
        
        if self.uncertainty:
#            self.sigma_net = nn.Sequential(
#                schnetpack2.nn.base.GetItem('representation'),
#                schnetpack2.custom.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation, bn=bn, p=p, sdr=sdr)
#            )
            self.sigma_act = torch.nn.functional.softplus#lambda x:x.abs()###
        
        # Make standardization separate
        self.standardize = schnetpack2.nn.base.ScaleShift(mean, stddev)

        if aggregation_mode == 'sum':
            self.atom_pool = schnetpack2.nn.base.Aggregate(axis=1, mean=False)
        elif aggregation_mode == 'avg':
            self.atom_pool = schnetpack2.nn.base.Aggregate(axis=1, mean=True)

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Structure.Z]
        atom_mask = inputs[Structure.atom_mask]

        yi = self.out_net(inputs)
        
        B, A = atom_mask.shape 
        
        yi_mean = yi.mean*self.standardize.stddev+self.standardize.mean
        yi_std  = yi.stddev*self.standardize.stddev
        
        if self.uncertainty:
#            yi = torch.cat([yi[:,:,:-1], self.sigma_act(yi[:,:,-1:])],dim=-1)
#            sigma = self.atom_pool(self.sigma_act(self.sigma_net(inputs))/10, atom_mask)  

            sigma = (self.atom_pool(self.sigma_act(yi[:,:,-1:])/10, atom_mask)+1e-5).sqrt()
            yi = yi[:,:,:-1]


        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi_mean + y0

        y = (yi_mean.view(B,A) * atom_mask).sum(-1)
        y_var  = ((yi_std+1e-5).pow(2).view(B,A) * atom_mask).sum(-1)
        
        y_dist = gpytorch.distributions.MultivariateNormal(y.unsqueeze(-1), y_var[...,None,None])

        result = {"y_dist": y_dist, "y": y, "natoms" : inputs['_atomic_numbers']}
        
        if self.uncertainty:
              
             result["sigma"] = sigma

        if self.return_contributions:
            result['yi'] = yi  

        return result


class Energy(Atomwise):
    """
        Predicts energy.

        Args:
            n_in (int): input dimension of representation
            pool_mode (str): one of {sum, avg} (default: sum)
            n_layers (int): number of nn in output network (default: 2)
            n_neurons (list of int or None): number of neurons in each layer of the output network.
                                              If `None`, divide neurons by 2 in each layer. (default: none)
            activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
            return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
            create_graph (bool): if True, graph of forces is created (default: False)
            return_force (bool): if True, forces will be calculated (default: False)
            mean (torch.FloatTensor): mean of energy (default: None)
            stddev (torch.FloatTensor): standard deviation of the energy (default: None)
            atomref (torch.Tensor): reference single-atom properties
            outnet (callable): network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                               not normalized. If set to None (default), a pyramidal network is generated automatically.

        Returns:
            tuple: Prediction for energy.

            If requires_dr is true additionally returns forces
        """

    def __init__(self, n_in, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=schnetpack2.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=False, train_embeddings=False, uncertainty=False, sdr=False, var_coeff=0.1,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None, bn=False, p=0, return_stress=False, return_hessian=False):
        super(Energy, self).__init__(n_in, 1, aggregation_mode, n_layers, n_neurons, activation,
                                     return_contributions, return_force, create_graph, mean, stddev,
                                     atomref, 100, outnet, train_embeddings, bn, p, uncertainty, sdr, var_coeff=var_coeff)
        self.return_stress  = return_stress
        self.return_hessian = return_hessian
        self.uncertainty    = uncertainty
        
    def forward(self, inputs):
        r"""
        predicts energy
        """
        result = super(Energy, self).forward(inputs)

        if self.requires_dr:
            if self.return_stress:
                forces = -grad(result["y"], inputs[Structure.R],
                           grad_outputs=torch.ones_like(result["y"]),
                           create_graph=True,retain_graph=True)[0]
                n_batch = inputs[Structure.R].size()[0]
                idx_m = torch.arange(n_batch, device=inputs[Structure.R].device, dtype=torch.long)[:,
                        None, None]

                # Subtract positions of central atoms to get distance vectors
                #B,A,N,C = dist_vec.shape
                #dist_vec = dist_vec.view(B,A*N,C)
                pair_force = grad(result['y'], inputs['dist_vec'],
                                  grad_outputs=torch.ones_like(inputs['dist_vec']),
                                  create_graph=False)[0]
                #result['stress'] = torch.sum(dist_vec.mm(pair_force.T),(1,2)) / 2.
                x_chol =  torch.cholesky(torch.einsum('bik,bjk->bij',inputs['_cell'],inputs['_cell']))
                V = x_chol[:,0,0]*x_chol[:,1,1]*x_chol[:,2,2]
                result['stress'] = -torch.einsum('abcd,abch->adh', inputs['dist_vec'], pair_force)/2./V*1.60217662e3
                if not self.training:
                    forces.sum().backward()
                    self.zero_grad()
                    
            elif self.return_hessian:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=True, retain_graph=True)[0]
                result['hessian'] = -grad(forces, inputs[Structure.R], create_graph=False)[0]
                if not self.training:
                    forces.sum().backward()
                    self.zero_grad()
            
            elif self.uncertainty:
                forces = -grad(result["y"], inputs[Structure.R], 
                                grad_outputs=torch.ones_like(result["y"]),
                                create_graph=True,retain_graph=True)[0]
                               
                forces_std   = -grad(result["sigma"], inputs[Structure.R],
                                   grad_outputs=torch.ones_like(result["y"]),
                                   create_graph=self.training,retain_graph=self.training)[0]
                
                result['sigma_forces'] = forces_std.abs()#nn.functional.relu(forces_std) 
                
            else:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=self.training,retain_graph=self.training)[0]
                                  
            result['dydx'] = forces

        return result

class MultiOutput(Atomwise):
    """
        Predicts energy.

        Args:
            n_in (int): input dimension of representation
            pool_mode (str): one of {sum, avg} (default: sum)
            n_layers (int): number of nn in output network (default: 2)
            n_neurons (list of int or None): number of neurons in each layer of the output network.
                                              If `None`, divide neurons by 2 in each layer. (default: none)
            activation (function): activation function for hidden nn (default: spk.nn.activations.shifted_softplus)
            return_contributions (bool): If True, latent atomic contributions are returned as well (default: False)
            create_graph (bool): if True, graph of forces is created (default: False)
            return_force (bool): if True, forces will be calculated (default: False)
            mean (torch.FloatTensor): mean of energy (default: None)
            stddev (torch.FloatTensor): standard deviation of the energy (default: None)
            atomref (torch.Tensor): reference single-atom properties
            outnet (callable): network used for atomistic outputs. Takes schnetpack input dictionary as input. Output is
                               not normalized. If set to None (default), a pyramidal network is generated automatically.

        Returns:
            tuple: Prediction for energy.

            If requires_dr is true additionally returns forces
        """

    def __init__(self, n_in, n_out, aggregation_mode='sum', n_layers=2, n_neurons=None,
                 activation=schnetpack2.nn.activations.shifted_softplus,
                 return_contributions=False, create_graph=False, train_embeddings=False, sdr = False, uncertainty=False,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None, bn=False, p=0, var_coeff=0.1,  return_stress=False, return_hessian=False):
        super(MultiOutput, self).__init__(n_in, n_out, aggregation_mode, n_layers, n_neurons, activation,
                                     return_contributions, return_force, create_graph, mean, stddev,
                                     atomref, 100, outnet, train_embeddings, bn, p, uncertainty, sdr, var_coeff=var_coeff)
        self.return_stress = return_stress
        self.return_hessian = return_hessian
        self.uncertainty = uncertainty

    def forward(self, inputs):
        r"""
        predicts energy
        """
        result = super(MultiOutput, self).forward(inputs)
        
#        if self.uncertainty_own:
#            result["sigma"] = torch.nn.functional.softplus(result["y"][:,1])
#            result["y"]     = result["y"][:,0]
            
        if self.requires_dr:
            if self.return_stress:
                forces = -grad(result["y"], inputs[Structure.R],
                           grad_outputs=torch.ones_like(result["y"]),
                           create_graph=self.create_graph,retain_graph=self.training)[0]
                n_batch = inputs[Structure.R].size()[0]
                idx_m = torch.arange(n_batch, device=inputs[Structure.R].device, dtype=torch.long)[:,
                        None, None]

                # Subtract positions of central atoms to get distance vectors
                #B,A,N,C = dist_vec.shape
                #dist_vec = dist_vec.view(B,A*N,C)
                pair_force = grad(result['y'], inputs['dist_vec'],
                                  grad_outputs=torch.ones_like(inputs['dist_vec']),
                                  create_graph=False)[0]
                #result['stress'] = torch.sum(dist_vec.mm(pair_force.T),(1,2)) / 2.
                x_chol =  torch.cholesky(torch.einsum('bik,bjk->bij',inputs['_cell'],inputs['_cell']))
                V = x_chol[:,0,0]*x_chol[:,1,1]*x_chol[:,2,2]
                result['stress'] = -torch.einsum('abcd,abch->adh', inputs['dist_vec'], pair_force)/2./V*1.60217662e3
            elif self.return_hessian:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=self.create_graph, retain_graph=self.training)[0]
                result['hessian'] = -grad(forces, inputs[Structure.R], create_graph=False)[0]
                
            elif self.uncertainty:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=True,retain_graph=True)[0]
                               
                forces_std   = -grad(result["sigma"], inputs[Structure.R],
                                   grad_outputs=torch.ones_like(result["y"]),
                                   create_graph=self.training,retain_graph=self.training)[0]
                
                result['sigma_forces'] = forces_std.abs()#nn.functional.relu(forces_std) 
            else:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=self.training,retain_graph=self.training)[0]
            result['dydx'] = forces

        return result