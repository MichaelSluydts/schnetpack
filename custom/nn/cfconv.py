import torch
from torch import nn as nn
from schnetpack2.custom.nn import Dense
from schnetpack2.custom.nn.base import Aggregate
from schnetpack2.custom.nn.layers import Linear_sdr, BN_drop_lin

import pdb

__all__ = [
    'CFConv', 'CFConvAngular', 'CFConvHop'
]

class CFConv(nn.Module):
    """
    Continuous-filter convolution layer for SchNet.

    Args:
        n_in (int): Number of input dimensions
        n_filters (int): Number of filter dimensions
        n_out (int): Number of output dimensions
        filter_network (nn.Module): Calculates filter
        cutoff_network (nn.Module): Calculates optional cutoff function (default: None)
        activation (function): Activation function
        normalize_filter (bool): If true, normalize filter to number of neighbors (default: false)
        axis (int): axis over which convolution should be applied
    """

    def __init__(self, n_in, n_filters, n_out, filter_network, cutoff_network=None,
                 activation=None, normalize_filter=False, sdr=False, bn=False, p=0.,var_coeff=0.1, axis=2):
        super(CFConv, self).__init__()
        if bn:
            self.in2f = BN_drop_lin(n_in, n_filters, bias=False, bn=True, p=p)
            self.f2out = BN_drop_lin(n_in, n_filters, bn=True, p=p)
        elif not sdr:
            self.in2f = Dense(n_in, n_filters, bias=False)
            self.f2out = Dense(n_filters, n_out, activation=activation)
        else:
            self.in2f = Linear_sdr(n_in, n_filters, bias=False, var_coeff=var_coeff)
            self.f2out = Linear_sdr(n_filters, n_out, activation=activation, var_coeff=var_coeff)  
                   
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Input representation of atomic environments.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            pairwise_mask (torch.Tensor): Mask to filter out non-existing neighbors introduced via padding.
            f_ij (torch.Tensor): Use at your own risk. Set to None otherwise.

        Returns:
            torch.Tensor: Continuous convolution.

        """
        if f_ij is None:
            f_ij = r_ij#.unsqueeze(-1)

        # calculate filter
        W = self.filter_network(f_ij)

        # apply optional cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij).squeeze(-1)
            #            print(C)
            W = W*C.unsqueeze(-1)

        # convolution
        y = self.in2f(x)

        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))

        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = y * W
        y = self.agg(y, pairwise_mask)

        y = self.f2out(y)

        return y
        
class CFConvAngular(nn.Module):
    """
    Continuous-filter convolution layer for SchNet.

    Args:
        n_in (int): Number of input dimensions
        n_filters (int): Number of filter dimensions
        n_out (int): Number of output dimensions
        filter_network (nn.Module): Calculates filter
        cutoff_network (nn.Module): Calculates optional cutoff function (default: None)
        activation (function): Activation function
        normalize_filter (bool): If true, normalize filter to number of neighbors (default: false)
        axis (int): axis over which convolution should be applied
    """

    def __init__(self, n_in, n_filters, n_out, filter_network, cutoff_network=None,
                 activation=None, normalize_filter=False, sdr=False, bn=False, p=0., axis=2):
        super(CFConvAngular, self).__init__()
        if bn:
            self.in2f = BN_drop_lin(n_in, n_filters, bias=False, bn=True, p=p)
            self.f2out = BN_drop_lin(n_in, n_filters, bn=True, p=p)
        elif not sdr:
            self.in2f = Dense(n_in, n_filters, bias=False)
            self.f2out = Dense(n_filters, n_out, activation=activation)
        else:
            self.in2f = Linear_sdr(n_in, n_filters, bias=False)
            self.f2out = Linear_sdr(n_filters, n_out, activation=activation)  
                   
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)

    def forward(self, x, r_ij, neighbors_j, neighbors_k, pairwise_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Input representation of atomic environments.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            pairwise_mask (torch.Tensor): Mask to filter out non-existing neighbors introduced via padding.
            f_ij (torch.Tensor): Use at your own risk. Set to None otherwise.

        Returns:
            torch.Tensor: Continuous convolution.

        """
        if f_ij is None:
            f_ij = r_ij#.unsqueeze(-1)

        # calculate filter
        W = self.filter_network(f_ij)

        # apply optional cutoff
#        if self.cutoff_network is not None:
#            C = self.cutoff_network(r_ij)
#            #            print(C)
#            W = W*C.unsqueeze(-1)

        # convolution
        y = self.in2f(x)

        nbh_size = neighbors_j.size()
        nbh_j = neighbors_j.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh_j = nbh_j.expand(-1, -1, y.size(2))

        nbh_k = neighbors_k.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh_k = nbh_k.expand(-1, -1, y.size(2))
        
        y_j = torch.gather(y, 1, nbh_j)
        y_j = y_j.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y_k = torch.gather(y, 1, nbh_k)
        y_k = y_k.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = y_j * W * y_k
        y = self.agg(y, pairwise_mask)

        y = self.f2out(y)

        return y
        
class CFConvHop(nn.Module):
    """
    Continuous-filter convolution layer for SchNet.

    Args:
        n_in (int): Number of input dimensions
        n_filters (int): Number of filter dimensions
        n_out (int): Number of output dimensions
        filter_network (nn.Module): Calculates filter
        cutoff_network (nn.Module): Calculates optional cutoff function (default: None)
        activation (function): Activation function
        normalize_filter (bool): If true, normalize filter to number of neighbors (default: false)
        axis (int): axis over which convolution should be applied
    """

    def __init__(self, n_in, n_filters, n_out, filter_network, cutoff_network=None,
                 activation=None, normalize_filter=False, sdr=False, bn=False, p=0.,var_coeff=0.1, axis=2, order=1):
        super(CFConvHop, self).__init__()
        if bn:
            self.in2f = BN_drop_lin(n_in, n_filters, bias=False, bn=True, p=p)
            self.f2out = BN_drop_lin(n_in, n_filters, bn=True, p=p)
        elif not sdr:
            self.in2f = Dense(n_in, n_filters, bias=False)
            self.f2out = Dense(n_filters, n_out, activation=activation)
        else:
            self.in2f = Linear_sdr(n_in, n_filters, bias=False, var_coeff=var_coeff)
            self.f2out = Linear_sdr(n_filters, n_out, activation=activation, var_coeff=var_coeff)  
                   
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)
        self.order = order
        self.cutoff = self.cutoff_network.cutoff.cuda()

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Input representation of atomic environments.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            pairwise_mask (torch.Tensor): Mask to filter out non-existing neighbors introduced via padding.
            f_ij (torch.Tensor): Use at your own risk. Set to None otherwise.

        Returns:
            torch.Tensor: Continuous convolution.

        """
#        pdb.set_trace()
        sim_mat = torch.exp(-5*r_ij/self.cutoff)
        
        sim_mat_temp = torch.zeros_like(sim_mat)
        sim_mat_temp[pairwise_mask != 0] = sim_mat[pairwise_mask != 0]
        sim_mat = sim_mat_temp
#        sim_mat[~pairwise_mask.byte()] = 0.0

        f_ij =  [sim_mat]
        
        N_atoms = pairwise_mask.sum(-1)
        N_atoms.clamp_(1,N_atoms.max())
        
        for i in range(1, self.order+1):
            f_ij.append(torch.bmm(f_ij[-1], sim_mat)/N_atoms[:,:,None])
        
        f_ij = torch.stack(f_ij, dim=-1)#torch.cat([a.unsqueeze(-1) for a in A], -1)
#        A = self.bns(A.transpose(1,3)).transpose(1,3)
        W = self.filter_network(f_ij)

        # apply optional cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij).squeeze(-1)
            W = W*C.unsqueeze(-1)*pairwise_mask[:,:,:,None]
            
#        W = W/W.sum(-2)[:,:,None,:]
#        W = W
 
        # convolution
        x = self.in2f(x)
        x = (W*x[:,:,None,:]).sum(-2)

        x = self.f2out(x)

        return x
