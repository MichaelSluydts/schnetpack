import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
cuda = torch.cuda.is_available()
import numpy as np
import schnetpack2
from schnetpack2.nn.base import Dense, Aggregate
from schnetpack2.custom.nn.layers import Linear_sdr, BN_drop_lin

class GATConv(nn.Module):

    def __init__(self, heads=1., concat = False, n_in = 128, n_out = 128, activation = schnetpack2.nn.activations.shifted_softplus, sdr = False, uncertainty = False):
        self.heads = torch.tensor(np.float(heads))
        self.concat = concat

        self.alpha = nn.Linear(self.heads,1)
        self.necks = schnetpack2.custom.nn.layers.Linear_sdr(n_in, n_out, self.heads, activation=activation,bias=False) if sdr else schnetpack2.nn.base.Dense(n_in, n_out, activation=activation,bias=False)

    def forward(self,x):
        pdb.set_trace()
        B, N, F = x.shape
        x = x.unsqueeze(0).repeat(self.heads,-1,-1)
        x = self.necks(x)
        x = self.alpha(x.permute(1,2,3,0))
        if self.concat:
            x = x.reshape(B,N,F*self.heads)
        else:
            x = x.mean()
        return x


class GATCFConv(nn.Module):
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
                 activation=None, normalize_filter=False, sdr=False, bn=False, p=0., axis=2,heads=4):
        super(GATCFConv, self).__init__()
        if bn:
            self.in2f = nn.ModuleList([BN_drop_lin(n_in, n_filters, bias=False, bn=True, p=p) for i in range(heads)])
            self.f2out = BN_drop_lin(n_in, n_filters, bn=True, p=p)
        elif not sdr:
            self.in2f = nn.ModuleList([Dense(n_in, n_filters, bias=False) for i in range(heads)])
            self.f2out = Dense(n_filters, n_out, activation=activation)
        else:
            self.in2f = nn.ModuleList([Linear_sdr(n_in, n_filters, bias=False) for i in range(heads)])
            self.f2out = Linear_sdr(n_filters, n_out, activation=activation)

        self.filter_network = filter_network
        self.cutoff_network = cutoff_network
        self.agg = Aggregate(axis=axis, mean=normalize_filter)
        self.heads = heads
        self.att = nn.ModuleList([Dense(2*n_filters, 1, bias = False) for i in range(heads)])

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
            f_ij = r_ij  # .unsqueeze(-1)

        # calculate filter
        W = self.filter_network(f_ij)

        # apply optional cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij).squeeze(-1)
            #            print(C)
            W = W * C.unsqueeze(-1)

        # convolution
        out = None

        for i in range(self.heads):
            y = self.in2f[i](x)
            B, N, F = y.shape

            nbh_size = neighbors.size()
            nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
            nbh = nbh.expand(-1, -1, y.size(2))
            yorig = y
            y = torch.gather(y, 1, nbh)
            e = torch.cat((yorig[:, :, None, :].repeat(1, 1, y.shape[1] // N, 1), y.view(B, N, -1, F)), dim=3).view(B,N,-1,2*F)

            att = self.att[i](e)
            alpha = nn.functional.softmax(schnetpack2.nn.activations.shifted_softplus(att),dim=2)
            alpha = alpha.reshape(B,-1,1).repeat(1,1,F)
            y = alpha*y

            y= y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

            if out is None:
                out = y * W
            else:
                out += y * W

        y = self.agg(out/self.heads, pairwise_mask)

        y = self.f2out(y)

        return y
