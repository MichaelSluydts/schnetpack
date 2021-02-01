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
import schnetpack2.custom.nn.layers
from schnetpack2.custom.nn.cutoff import PhysNetCutoff

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
                 train_embeddings=False, bn=False, p=0, uncertainty=False, sdr=False, var_coeff=0.1, num_inducing_points = None):
        super(Atomwise, self).__init__(n_in, n_out, requires_dr)

        self.n_layers = n_layers
        self.create_graph = create_graph
        self.return_contributions = return_contributions
        self.num_inducing_points = num_inducing_points

        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev
        self.register_buffer('mean', mean)
        self.register_buffer('stddev', stddev)
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


        if outnet is None and num_inducing_points is None:
            self.out_net = nn.Sequential(
                schnetpack2.nn.base.GetItem('representation'),
                schnetpack2.custom.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation, bn=bn, p=p, sdr=sdr, var_coeff=var_coeff)
            )
        elif isinstance(num_inducing_points, int):
            out_net = schnetpack2.custom.nn.blocks.MLP(num_inducing_points, n_out, n_neurons, n_layers, activation, bn=bn, p=p, sdr=sdr, var_coeff=var_coeff)
            self.out_net = schnetpack2.custom.nn.layers.InducingPointBlock(n_in, num_inducing_points, out_net)
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
        self.standardize = schnetpack2.custom.nn.base.ScaleShift(mean, stddev)

        if aggregation_mode == 'sum':
            self.atom_pool = schnetpack2.custom.nn.base.Aggregate(axis=1, mean=False,start=1.)
        elif aggregation_mode == 'avg':
            self.atom_pool = schnetpack2.custom.nn.base.Aggregate(axis=1, mean=True,start=1.)

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Structure.Z]
        atom_mask = inputs[Structure.atom_mask]
        result = {"natoms" : inputs['_atomic_numbers']}


        if not self.num_inducing_points:
            yi = self.out_net(inputs)
        else:
            yi, var_i,gamma = self.out_net(inputs)
            yi = self.standardize(yi)
            var = (var_i * atom_mask).sum(1, keepdim=True)*self.standardize.stddev.pow(2)
            result["gamma"] = gamma

        if self.uncertainty:
#            yi = torch.cat([yi[:,:,:-1], self.sigma_act(yi[:,:,-1:])],dim=-1)
#            sigma = self.atom_pool(self.sigma_act(self.sigma_net(inputs))/10, atom_mask)  

            sigma = (self.atom_pool(self.sigma_act(yi[:,:,-1:])/10, atom_mask)+1e-5).sqrt()
            yi = yi[:,:,:-1]


        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi #+ y0

        y = self.atom_pool(yi, atom_mask)
        #y = self.standardize(y)

        result["y"] = y
        result['yout'] = self.standardize(y.float()) #+ torch.sum(y0.float(),1)
        if self.uncertainty:
             result["sigma"] = sigma

        if self.return_contributions:
            result['yi'] = yi  
            
        if self.num_inducing_points:
            result['sigma'] = (var+1e-5).sqrt()  

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
                 return_contributions=False, create_graph=False, train_embeddings=False, uncertainty=False, sdr=False, var_coeff=0.1, num_inducing_points = None,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None, bn=False, p=0, return_stress=False, return_hessian=False):
        super(Energy, self).__init__(n_in, 1, aggregation_mode, n_layers, n_neurons, activation,
                                     return_contributions, return_force, create_graph, mean, stddev,
                                     atomref, 100, outnet, train_embeddings, bn, p, uncertainty, sdr, var_coeff=var_coeff, num_inducing_points=num_inducing_points)
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
            
            elif "sigma" in result:
                forces = -grad(result["y"], inputs[Structure.R], 
                                grad_outputs=torch.ones_like(result["y"]),
                                create_graph=True,retain_graph=True)[0]
                               
#                forces_std   = -grad(result["sigma"], inputs[Structure.R],
#                                   grad_outputs=torch.ones_like(result["y"]),
#                                   create_graph=self.training,retain_graph=self.training)[0]
#                
#                result['sigma_forces'] = forces_std.abs()#nn.functional.relu(forces_std) 
                
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
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None, bn=False, p=0, var_coeff=0.1,  return_stress=False, return_hessian=False,repulsive=0.,per_atom=False):
        super(MultiOutput, self).__init__(n_in, n_out, aggregation_mode, n_layers, n_neurons, activation,
                                     return_contributions, return_force, create_graph, mean, stddev,
                                     atomref, 100, outnet, train_embeddings, bn, p, uncertainty, sdr, var_coeff=var_coeff)
        self.return_stress = return_stress
        self.return_hessian = return_hessian
        self.uncertainty = uncertainty
        self.repulsive=repulsive
        self.per_atom = per_atom

    def forward(self, inputs):
        r"""
        predicts energy
        """
        result = super(MultiOutput, self).forward(inputs)
        
        if self.repulsive > 0.:
             corr = repulsive_correction(inputs,cutoff=self.repulsive)
             if corr > 0:
                 print('----REPEL----' + str(corr) + '-------------')
             result['y'] =  result['y'] + corr.type(result['y'].dtype)
#        if self.uncertainty_own:
#            result["sigma"] = torch.nn.functional.softplus(result["y"][:,1])
#            result["y"]     = result["y"][:,0]
            
        if self.requires_dr:
            if self.return_stress:
                forces = -grad(result["yout"], inputs[Structure.R],
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
            if self.per_atom:
                forces = forces *torch.sum((inputs['_atomic_numbers'] != 0), 1, keepdim=True).unsqueeze(-1).type(forces.dtype)
            result['dydx'] = self.stddev.type(forces.dtype)*forces

        return result

class EnergyCoulomb(Atomwise):
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
                 return_contributions=False, create_graph=False, train_embeddings=False,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, outnet=None, bn=False, p=0, return_stress=False, return_hessian=False):
        super(EnergyCoulomb, self).__init__(n_in, 1, aggregation_mode, n_layers, n_neurons, activation,
                                     return_contributions, return_force, create_graph, mean, stddev,
                                     atomref, 100, outnet, train_embeddings, bn, p)
                                     
        self.return_stress = return_stress
        self.return_hessian = return_hessian
        self.charge_net = nn.Sequential(
                        schnetpack2.nn.base.Dense(n_in, n_in//2,
                                                 activation=schnetpack2.nn.activations.shifted_softplus),
                        schnetpack2.nn.base.Dense(n_in//2, 1, activation=None)
                    )
#        self.electrostatic_constant = nn.Parameter(1.0)

    def forward(self, inputs):
        r"""
        predicts energy
        """
#        pdb.set_trace()
        result = super(EnergyCoulomb, self).forward(inputs)
        distances = torch.norm((inputs[Structure.R][:,None,:,:]-inputs[Structure.R][:,:,None,:]),dim=-1)
               
        _atom_mask_3d = (inputs[Structure.atom_mask][:,None,:]*inputs[Structure.atom_mask][:,:,None])        
        diag_mask = 1-torch.eye(distances.shape[-1]).unsqueeze(0).expand(distances.shape[0],-1,-1).cuda()
        tot_mask = _atom_mask_3d*diag_mask
        
        distances_tmp = torch.zeros_like(distances)
        distances_tmp[tot_mask != 0] = distances[tot_mask != 0]
        distances = distances_tmp        
        
        charges = self.charge_net(inputs['representation'])        
        coulomb_matrix = charges[:,:,0][:,None,:]*1.0*(1e-5+distances).pow(-2)*charges

#        coulomb_matrix[diag_mask] = 0.0
#        coulomb_matrix[_atom_mask_3d==0] = 0.0
  
        coulomb_matrix_tmp = torch.zeros_like(coulomb_matrix)
        coulomb_matrix_tmp[tot_mask != 0] = coulomb_matrix[tot_mask != 0]
        coulomb_matrix = coulomb_matrix_tmp
        
        electrostatic_energy = coulomb_matrix.sum((1,2))#(coulomb_matrix*(~diag_mask).float().cuda()*(1-_atom_mask_3d).float().cuda()).sum()/2
        
        result['y'] =  result['y'] + electrostatic_energy
        
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
            else:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=self.training,retain_graph=self.training)[0]
            result['dydx'] = forces

        return result
        
class CoulombApproximation(nn.Module):
    def __init__(self, cutoff):
        super(CoulombApproximation, self).__init__()
        self.cutoff_function = PhysNetCutoff(cutoff)
        
    def forward(self, r_ij):
        return self.cutoff_function(2*r_ij)/(r_ij.pow(2)+1).sqrt()+(1-self.cutoff_function(2*r_ij))/(r_ij + 1e-6)

class OutputPhysNet(nn.Module):
    def __init__(self, aggregation_mode='sum',
                 create_graph=False, train_embeddings=False, uncertainty=False, sdr=False, var_coeff=0.1, cutoff = 5.0,
                 return_force=False, mean=None, stddev=None, atomref=None, max_z=100, bn=False, p=0, return_stress=False, return_hessian=False):
        super(OutputPhysNet, self).__init__()
        self.requires_dr  = return_force
        self.create_graph = create_graph

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
        
        self.uncertainty = uncertainty
        
        if self.uncertainty:
#            self.sigma_net = nn.Sequential(
#                schnetpack2.nn.base.GetItem('representation'),
#                schnetpack2.custom.nn.blocks.MLP(n_in, n_out, n_neurons, n_layers, activation, bn=bn, p=p, sdr=sdr)
#            )
            self.sigma_act = torch.nn.functional.softplus#lambda x:x.abs()###
        
        # Make standardization separate
        self.standardize = schnetpack2.nn.base.ScaleShift(mean, stddev)
        
        self.atom_pool = schnetpack2.custom.nn.base.Aggregate(axis=1, mean=False)
        
        self.return_stress  = return_stress
        self.return_hessian = return_hessian
        self.uncertainty    = uncertainty
        
        self.coulomb_approx = CoulombApproximation(cutoff)
        
    def forward(self, inputs):
        r"""
        predicts energy
        """
        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        atom_mask = inputs[Structure.atom_mask][:, :, None]

        Ei = self.standardize(inputs["representation"]["energies"])

        B, A, _ = Ei.shape        
#        if self.uncertainty:
#            sigma = (self.atom_pool(self.sigma_act(yi[:,:,-1:])/10, atom_mask)+1e-5).sqrt()
#            yi = yi[:,:,:-1]

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            Ei = Ei + y0

        E = (Ei*atom_mask).sum(1)#self.atom_pool(Ei, atom_mask)
        
        charges_masked = inputs["representation"]["charges"] * atom_mask
        Q = charges_masked.sum(1)
        N = torch.sum(atom_mask, 1, keepdim=True)
        
        q_eff  = (charges_masked - Q[...,None]/N) * atom_mask
        q_eff2 = q_eff[:,None,:,0] 
        
        diagonal_mask = torch.torch.eye(A).cuda().expand(B, A, A)
        interactions  = q_eff*q_eff2
        interactions  = interactions[diagonal_mask != 1].view(B, A, A - 1)
        
        dp_i = positions * q_eff 
        dp   = dp_i.sum(1)

        E = E + k_e/2*(self.coulomb_approx(inputs["representation"]["distances"])*interactions).sum()

        result = {"y": E, "natoms" : inputs['_atomic_numbers'], "Ei" : inputs["representation"]["energies"]* atom_mask, "Q" : Q, 'q' :  q_eff, "qi" : charges_masked, "dipole" : dp, "atom_mask":atom_mask, "coupling_constant": inputs["representation"]["coupling_constant"]}
        
        if self.requires_dr:
            if self.return_stress:
                forces = -grad(result["y"], inputs[Structure.R],
                           grad_outputs=torch.ones_like(result["y"]),
                           create_graph=True,retain_graph=True)[0]
                n_batch = inputs[Structure.R].size()[0]
                idx_m = torch.arange(n_batch, device=inputs[Structure.R].device, dtype=torch.long)[:,
                        None, None]

                pair_force = grad(result['y'], inputs['dist_vec'],
                                  grad_outputs=torch.ones_like(inputs['dist_vec']),
                                  create_graph=False)[0]

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
                                   grad_outputs=torch.ones_like(result["energy"]),
                                   create_graph=self.training,retain_graph=self.training)[0]
                
                result['sigma_forces'] = forces_std.abs()#nn.functional.relu(forces_std) 
                
            else:
                forces = -grad(result["y"], inputs[Structure.R],
                               grad_outputs=torch.ones_like(result["y"]),
                               create_graph=self.training,retain_graph=self.training)[0]
                                  
            result['dydx'] = forces

        return result

class DipoleMoment(Atomwise):
    """
    Predicts latent partial charges and calculates dipole moment.

    Args:
        n_in (int): input dimension of representation
        n_layers (int): number of layers in output network (default: 2)
        n_neurons (list of int or None): number of neurons in each layer of the output network.
                                          If `None`, divide neurons by 2 in each layer. (default: none)
        activation (torch.Function): activation function for hidden nn (default: schnetpack2.nn.activations.shifted_softplus)
        return_charges (bool): if True, latent atomic charges are returned as well (default: False)
        requires_dr (bool): set True, if derivative w.r.t. atom positions is required (default: False)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment instead of the vector (default: False)
        mean (torch.FloatTensor): mean of dipole (default: 0.0)
        stddev (torch.FloatTensor): stddev of dipole (default: 0.0)


    Returns:
        tuple: vector for the dipole moment

        If predict_magnitude is true returns the magnitude of the dipole moment instead of the vector

        If return_charges is true returns either vector or magnitude of the dipole moment, and latent atomic charges

        If requires_dr is true returns derivative w.r.t. atom positions
    """

    def __init__(self, n_in, n_layers=2, n_neurons=None, activation=schnetpack2.nn.activations.shifted_softplus,
                 return_charges=False, requires_dr=False, outnet=None, predict_magnitude=False,
                 mean=torch.FloatTensor([0.0]), stddev=torch.FloatTensor([1.0])):
        self.return_charges = return_charges
        self.predict_magnitude = predict_magnitude
        super(DipoleMoment, self).__init__(n_in, 1, 'sum', n_layers, n_neurons, activation=activation,
                                           requires_dr=requires_dr, mean=mean, stddev=stddev, outnet=outnet)

    def forward(self, inputs):
        """
        predicts dipole moment
        """
        positions = inputs[Structure.R]
        atom_mask = inputs[Structure.atom_mask][:, :, None]

        charges = self.out_net(inputs)
        yi = positions * charges * atom_mask
        y = self.atom_pool(yi)

        if self.predict_magnitude:
            result = {"y": torch.norm(y, dim=1, keepdim=True)}
        else:
            result = {"y": y}

        if self.return_charges:
            result['yi'] = yi

        return result


class ElementalAtomwise(Atomwise):
    """
    Predicts properties in atom-wise fashion using a separate network for every chemical element of the central atom.
    Particularly useful for networks of Behler-Parrinello type.
    """

    def __init__(self, n_in, n_out=1, aggregation_mode='sum', n_layers=3, requires_dr=False, create_graph=False,
                 elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                 activation=schnetpack2.nn.activations.shifted_softplus,
                 return_contributions=False, mean=None, stddev=None, atomref=None, max_z=100):
        outnet = schnetpack2.nn.blocks.GatedNetwork(n_in, n_out, elements, n_hidden=n_hidden, n_layers=n_layers,
                                                   activation=activation)

        super(ElementalAtomwise, self).__init__(n_in, n_out, aggregation_mode, n_layers, None, activation,
                                                return_contributions, requires_dr, create_graph, mean, stddev, atomref,
                                                max_z, outnet)


class ElementalEnergy(Energy):
    """
    Predicts atom-wise contributions, but uses one separate network for every chemical element of
    the central atoms.
    Particularly useful for networks of Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of target property (default: 1)
        pool_mode (str): one of {sum, avg} (default: sum)
        n_layers (int): number of layers in output network (default: 3)
        return_force (bool): True, if derivative w.r.t. atom positions is required (default: False)
        elements (set of int): List of atomic numbers present in the training set {1,6,7,8,9} for QM9. (default: (1,6,7,8,9)
        n_hidden (int): number of neurons in each hidden layer of the output network. (default: 50)
        activation (torch.Function): activation function for hidden nn (default: schnetpack2.nn.activations.shifted_softplus)
        return_contributions (bool): If True, latent atomic con tributions are returned as well (default: False)
        mean (torch.FloatTensor): mean of energy
        stddev (torch.FloatTensor): standard deviation of energy
        atomref (torch.tensor): reference single-atom properties
        max_z (int): ignored if atomref is not learned (default: 100)
    """

    def __init__(self, n_in, n_out=1, aggregation_mode='sum', n_layers=3, return_force=False, create_graph=False,
                 elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                 activation=schnetpack2.nn.activations.shifted_softplus, return_contributions=False, mean=None,
                 stddev=None, atomref=None, max_z=100):
        outnet = schnetpack2.nn.blocks.GatedNetwork(n_in, n_out, elements, n_hidden=n_hidden, n_layers=n_layers,
                                                   activation=activation)

        super(ElementalEnergy, self).__init__(n_in, aggregation_mode, n_layers, None, activation, return_contributions,
                                              create_graph=create_graph, return_force=return_force, mean=mean,
                                              stddev=stddev, atomref=atomref, max_z=max_z, outnet=outnet)


class ElementalDipoleMoment(DipoleMoment):
    """
    Predicts partial charges and computes dipole moment using serparate NNs for every different element.
    Particularly useful for networks of Behler-Parrinello type.

    Args:
        n_in (int): input dimension of representation
        n_out (int): output dimension of representation (default: 1)
        n_layers (int): number of layers in output network (default: 3)
        return_charges (bool): If True, latent atomic charges are returned as well (default: False)
        requires_dr (bool): Set True, if derivative w.r.t. atom positions is required (default: False)
        predict_magnitude (bool): if True, predict the magnitude of the dipole moment instead of the vector (default: False)
        elements (set of int): List of atomic numbers present in the training set {1,6,7,8,9} for QM9. (default: frozenset(1,6,7,8,9))
        n_hidden (int): number of neurons in each layer of the output network. (default: 50)
        activation (function): activation function for hidden nn (default: schnetpack2.nn.activations.shifted_softplus)
        activation (function): activation function for hidden nn
        mean (torch.FloatTensor): mean of energy
        stddev (torch.FloatTensor): standard deviation of energy
    """

    def __init__(self, n_in, n_out=1, n_layers=3, return_charges=False, requires_dr=False, predict_magnitude=False,
                 elements=frozenset((1, 6, 7, 8, 9)), n_hidden=50,
                 activation=schnetpack2.nn.activations.shifted_softplus,
                 mean=None, stddev=None):
        outnet = schnetpack2.nn.blocks.GatedNetwork(n_in, n_out, elements, n_hidden=n_hidden, n_layers=n_layers,
                                                   activation=activation)

        super(ElementalDipoleMoment, self).__init__(n_in, n_layers, None, activation, return_charges, requires_dr,
                                                    outnet, predict_magnitude, mean, stddev)
def repulsive_correction(inputs, cutoff = 2.0):
   n_batch = inputs[Structure.R].size()[0]
   idx_m = torch.arange(n_batch, device=inputs[Structure.R].device, dtype=torch.long)[:,
           None, None]
   cell = inputs[Structure.cell]
   cell_offsets = inputs[Structure.cell_offset]
   neighbors = inputs[Structure.neighbors]
   neighbor_mask = inputs[Structure.neighbor_mask]
   distances = (inputs[Structure.R][idx_m, neighbors[:, :, :], :] - inputs[Structure.R][:, :, None, :])
   if cell is not None:
       B, A, N, D = cell_offsets.size()
       cell_offsets = cell_offsets.view(B, A * N, D)
       offsets = cell_offsets.bmm(cell)
       offsets = offsets.view(B, A, N, D)
       distances += offsets
   # Compute vector lengths
   distances = torch.norm(distances, 2, 3)
   if neighbor_mask is not None:
       # Avoid problems with zero distances in forces (instability of square root derivative at 0)
       # This way is neccessary, as gradients do not work with inplace operations, such as e.g.
       # -> distances[mask==0] = 0.0
       tmp_distances = torch.zeros_like(distances)
       tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
       distances = tmp_distances
   #_atom_mask_3d = (inputs[Structure.atom_mask][:, None, :] * inputs[Structure.atom_mask][:, :, None])
   #diag_mask = 1 - torch.eye(distances.shape[-1]).unsqueeze(0).expand(distances.shape[0], -1, -1).cuda()
   #tot_mask = _atom_mask_3d * diag_mask
   #tot_mask = diag_mask
   #tot_mask = torch.eye(distances.shape[-1]).unsqueeze(0).expand(distances.shape[0], distances.shape[1], -1).cuda()
   #distances_tmp = torch.zeros_like(distances)
   #distances_tmp[tot_mask != 0] = distances[tot_mask != 0]
   #distances = distances_tmp
   power = -5.
   matrix = (1e-5 + distances/cutoff).pow(power) -1.
   #        coulomb_matrix[diag_mask] = 0.0
   #        coulomb_matrix[_atom_mask_3d==0] = 0.0
   #matrix_tmp = torch.zeros_like(matrix)
   #matrix_tmp[tot_mask != 0] = matrix[tot_mask != 0]
   #matrix = matrix_tmp
   corr = matrix.to(device=inputs[Structure.R].device)*(distances < cutoff).float()*(distances > 0.).float()
   #corr = (matrix - torch.tensor(cutoff).pow(power).to(device=inputs[Structure.R].device))*(distances < cutoff).float()*(distances > 0.).float()
   #corr = (torch.exp(-distances)-torch.exp(-torch.tensor(cutoff).cuda()))*(distances < cutoff).float()
   corr = corr.sum((1, 2))
   return corr

