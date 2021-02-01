import torch.nn as nn
import torch
import schnetpack2.nn.acsf
import schnetpack2.nn.activations
import schnetpack2.nn.base
import schnetpack2.custom.nn.neighbors
import schnetpack2.nn.cutoff
from schnetpack2.custom.nn.cutoff import PhysNetCutoff
from schnetpack2.custom.data import Structure
from schnetpack2.custom.nn.layers import Linear_sdr
from schnetpack2.custom.nn.blocks import PreActivationResidualNN, OutputModuleBlock
from schnetpack2.custom.nn.acsf import GaussianSmearing, LaguerreSmearing, ChebyshevSmearing
import schnetpack2.custom.nn.cfconv
from functools import partial
import pdb
#from torch_geometric.nn import GATConv

class InteractionBlock(nn.Module):
    """
    SchNet interaction block for modeling quantum interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_basis, n_spatial_basis, activation = schnetpack2.nn.activations.shifted_softplus, n_layers=3,
                     normalize_filter=False, sdr=False, bn=False, p=0, var_coeff=0.1, axis=2):
        super(InteractionBlock, self).__init__()
        
        self.activation = activation
        if bn:
            self.in2f  = BN_drop_lin(n_atom_basis, n_atom_basis, bn=bn, p=p, actn = activation)
            self.dense = BN_drop_lin(n_atom_basis, n_atom_basis, bn=bn, p=p, actn = None)
        elif not sdr:
            self.in2f  = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, activation=activation)
            self.dense = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, activation=None)
        else:
            self.in2f  = Linear_sdr(n_atom_basis, n_atom_basis, activation=activation, var_coeff=var_coeff)
            self.dense = Linear_sdr(n_atom_basis, n_atom_basis, activation=None, var_coeff=var_coeff)            
        # initialize attention mask
        self.G = nn.Linear(n_spatial_basis, n_atom_basis, bias=False)

        self.residual_block = PreActivationResidualNN(n_atom_basis, n_atom_basis, n_hidden=None, n_layers=n_layers, activation=activation, sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)
            
        self.agg = schnetpack2.nn.base.Aggregate(axis=axis, mean=normalize_filter)
        
        self.mask = nn.Parameter(torch.ones(n_atom_basis))

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Atom-wise input representations.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            neighbor_mask (torch.Tensor): Mask to indicate virtual neighbors introduced via zeros padding.
            f_ij (torch.Tensor): Use at your own risk.

        Returns:
            torch.Tensor: SchNet representation.
        """
        y = self.activation(x)
        y = self.in2f(y)
        W = self.G(f_ij)
        
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))

        y2 = torch.gather(y, 1, nbh)
        y2 = y2.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y2 = y2 * W
        y2 = self.agg(y2, neighbor_mask)
        
        y = y+y2
        
        y = self.residual_block(y)
        y = self.activation(y)
        y = self.dense(y)
        
        return y + self.mask*x
        
class PhysnetModule(nn.Module):
    def __init__(self, n_atom_basis, n_spatial_basis, n_out=2, activation = schnetpack2.nn.activations.shifted_softplus, n_layers_interaction=3, n_layers_res = 2,
                     normalize_filter=False, sdr=False, bn=False, p=0, var_coeff=0.1):
        super(PhysnetModule, self).__init__()
        self.interaction_block = InteractionBlock(n_atom_basis, n_spatial_basis, activation = activation, n_layers=n_layers_interaction,
                                                         normalize_filter=normalize_filter, sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)
                                                         
        self.residual_block = PreActivationResidualNN(n_atom_basis, n_atom_basis, n_hidden=None, n_layers=n_layers_res, activation=activation, sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)
        
        self.output_module  = OutputModuleBlock(n_atom_basis, n_out, n_hidden=None, n_layers=n_layers_res, activation=activation, sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)
        
        self.output_model2  = OutputModuleBlock(n_atom_basis, 1, n_hidden=None, n_layers=n_layers_res, activation=activation, sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)
        
    def forward(self, x, r_ij, neighbors, neighbor_mask, atom_index1, atom_index2, f_ij=None):
        x = self.residual_block(self.interaction_block(x, r_ij, neighbors, neighbor_mask, f_ij))

        atom_index1 = atom_index1[...,None].expand(-1, -1, x.size(2))
        atom_index2 = atom_index2[...,None].expand(-1, -1, x.size(2))
        
        y1 = torch.gather(x, 1, atom_index1)
        y2 = torch.gather(x, 1, atom_index2)
        
        couplings = self.output_model2((y1-y2).abs()).squeeze(-1)
        
        return x, self.output_module(x), couplings
        

def get_expansions(expansion_type, n_filters, n_expansion_coeffs, max_val=5.0, start =0.0, trainable=True):    
    if expansion_type=='laguerre':
        return LaguerreSmearing(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs)   
    elif expansion_type=='chebyshev':
        return ChebyshevSmearing(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs, cutoff=max_val)
    else:# expansion_type=='gaussian':
        return GaussianSmearing(start, max_val, n_gaussians=n_filters, trainable=trainable)
        
class Exponential(nn.Module):
    def __init__(self):
        super(Exponential, self).__init__()
        
    def forward(self, r_ij):
        return torch.exp(-r_ij)
        
class PhysNet(nn.Module):
    """
    SchNet architecture for learning representations of atomistic systems
    as described in [#schnet1]_ [#schnet2]_ [#schnet3]_

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_filters (int): number of filters used in continuous-filter convolution
        n_interactions (int): number of interaction blocks
        cutoff (float): cutoff radius of filters
        n_gaussians (int): number of Gaussians which are used to expand atom distances
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
        coupled_interactions (bool): if true, share the weights across interaction blocks
            and filter-generating networks.
        return_intermediate (bool): if true, also return intermediate feature representations
            after each interaction block
        max_z (int): maximum allowed nuclear charge in dataset. This determines the size of the embedding matrix.

    References
    ----------
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet2] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    """

    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=1, cutoff=5.0, n_gaussians=25,
                 normalize_filter=False, coupled_interactions=False,
                 max_z=100, cutoff_network=None, pre_expansion_function =  Exponential(), trainable_gaussians=False, start=0.0,
                 distance_expansion=None, debug=False, expansion_type="gaussian", n_expansion_coeffs=10, sdr=False, bn=False, p=0, var_coeff=0.1, attention=0, return_stress=False):
        super(PhysNet, self).__init__()
        self.return_stress = return_stress
        self.debug = debug

        self.cutoff = cutoff

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # spatial features
        self.distances = schnetpack2.custom.nn.neighbors.AtomDistances(return_vecs=return_stress)
        
        self.pre_expansion_function = pre_expansion_function
        
        if cutoff_network is None:
            self.cutoff_network = PhysNetCutoff(cutoff)
        else:
            self.cutoff_network = cutoff_network
            
        if distance_expansion is None:
            self.distance_expansion = get_expansions(expansion_type, n_filters = n_filters, n_expansion_coeffs = n_expansion_coeffs, max_val=1.0, start = start, trainable=True)
        else:
            self.distance_expansion = distance_expansion

        # interaction network
        if isinstance(n_interactions,list):
            temp = []
            tempint = 0
            for i in n_interactions:
                temp += i*[PhysnetModule(n_atom_basis=n_atom_basis,
                                         n_spatial_basis=n_filters,
                                         n_out=2, n_layers_interaction=3, n_layers_res = 2,
                                         normalize_filter=normalize_filter,
                                         sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)]
                tempint += i
            n_interactions = tempint
            self.interactions = nn.ModuleList(temp)
        elif coupled_interactions:
            self.interactions = nn.ModuleList([
                                                  PhysnetModule(n_atom_basis=n_atom_basis,
                                                                n_spatial_basis=n_filters,
                                                                n_out=2, n_layers_interaction=3, n_layers_res = 2,
                                                                normalize_filter=normalize_filter,
                                                                sdr=sdr, bn=bn, p=p, var_coeff=var_coeff)
                                              ] * n_interactions)
        else:
            interaction_block_list = []
            for _ in range(n_interactions):
                interaction_block_list.append(PhysnetModule(n_atom_basis=n_atom_basis,
                                                            n_spatial_basis=n_filters,
                                                            n_out=2, n_layers_interaction=3, n_layers_res = 2,
                                                            normalize_filter=normalize_filter,
                                                            sdr=sdr, bn=bn, p=p, var_coeff=var_coeff))
            
            self.interactions = nn.ModuleList(interaction_block_list)

    def forward(self, inputs):
        if self.debug:
            pdb.set_trace()
        """
        Args:
            inputs (dict of torch.Tensor): SchNetPack format dictionary of input tensors.

        Returns:
            torch.Tensor: Final Atom-wise SchNet representation.
            torch.Tensor: Atom-wise SchNet representation of intermediate layers.
        """
        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        cell = inputs[Structure.cell]
        cell_offset = inputs[Structure.cell_offset]
        neighbors = inputs[Structure.neighbors]
        neighbor_mask = inputs[Structure.neighbor_mask]
        atom_index_1 = inputs["atom_index_0"].long()
        atom_index_2 = inputs["atom_index_1"].long()
        
        # atom embedding
        x = self.embedding(atomic_numbers)

        # spatial features

        if Structure.r_ij not in inputs.keys():
            if self.return_stress:
                r_ij, dist_vec = self.distances(positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask)
            else:
                r_ij = self.distances(positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask)
        else:
            r_ij = inputs[Structure.r_ij]
            
        f_ij = self.cutoff_network(r_ij)[:,:,:,None]*self.distance_expansion(self.pre_expansion_function(r_ij))

        outs = 0.0
        couplings = 0.0

        for interaction in self.interactions:
            v, out, coupling = interaction(x, r_ij, neighbors, neighbor_mask, atom_index_1, atom_index_2, f_ij=f_ij)
            
            x = x + v
            
            outs = out + outs
            
            couplings = coupling + couplings
  
        couplings = couplings*((atom_index_1+atom_index_2)>0).float()
            
        outs = outs/100
            
        E, Q = outs[:,:,:1], outs[:,:,1:]
        
        results = {"positions":positions, "neigbor_mask":neighbor_mask, "energies": E, "charges": Q, "coupling_constant": couplings, "distances":r_ij}    
        #results = {"positions":positions, "neigbor_mask":neighbor_mask, "energies": E, "charges": Q, "coupling_constant": couplings, "distances":r_ij}    
        
        return results
