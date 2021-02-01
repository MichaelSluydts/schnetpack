import torch.nn as nn
import torch
import schnetpack2.nn.acsf
import schnetpack2.nn.activations
import schnetpack2.nn.base
import schnetpack2.custom.nn.neighbors
import schnetpack2.nn.cutoff
from schnetpack2.custom.data import Structure
from schnetpack2.custom.nn.layers import Linear_sdr
from schnetpack2.custom.nn.acsf import GaussianSmearing, LaguerreSmearing, ChebyshevSmearing
import schnetpack2.custom.nn.cfconv
import pdb
#from torch_geometric.nn import GATConv

class SchNetInteraction(nn.Module):
    """
    SchNet interaction block for modeling quantum interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_basis, n_spatial_basis, n_filters,
                 normalize_filter=False,cutoff=5.0, cutoff_network=schnetpack2.nn.cutoff.HardCutoff, sdr=False, bn=False, filter_network=None, p=0, var_coeff=0.1, attention=0):
        super(SchNetInteraction, self).__init__()

        # initialize filters
        if not filter_network:
            if not sdr:
                if not bn:
                    self.filter_network = nn.Sequential(
                        schnetpack2.nn.base.Dense(n_spatial_basis, n_filters,
                                                 activation=schnetpack2.nn.activations.shifted_softplus),
                        schnetpack2.nn.base.Dense(n_filters, n_filters)
                    )
                    #original: nn.Sequential(schnetpack2.nn.base.Dense(n_spatial_basis, n_filters,activation=schnetpack2.nn.activations.shifted_softplus),schnetpack2.nn.base.Dense(n_filters, n_filters))
                else:
                    self.filter_network = nn.Sequential(
                        schnetpack2.custom.nn.layers.BN_drop_lin(n_spatial_basis, n_filters,
                                                 actn=schnetpack2.nn.activations.shifted_softplus),
                        schnetpack2.custom.nn.layers.BN_drop_lin(n_filters, n_filters, actn=None)
                    )

            else:
                self.filter_network = nn.Sequential(
                    Linear_sdr(n_spatial_basis, n_filters,
                                             activation=schnetpack2.nn.activations.shifted_softplus, var_coeff=var_coeff),
                    Linear_sdr(n_filters, n_filters, var_coeff=var_coeff)
                ) 
        else:
            self.filter_network = filter_network              

        self.cutoff_network = cutoff_network(cutoff)

        # initialize interaction blocks
        if attention > 0:
            self.cfconv = schnetpack2.custom.nn.attention.GATCFConv(n_atom_basis, n_filters, n_atom_basis,
                                                             self.filter_network,
                                                             cutoff_network=self.cutoff_network,
                                                             activation=schnetpack2.nn.activations.shifted_softplus,
                                                             normalize_filter=normalize_filter,
                                                             sdr=sdr, bn=bn, p=p,heads=attention)
        else:
            self.cfconv = schnetpack2.custom.nn.cfconv.CFConv(n_atom_basis, n_filters, n_atom_basis,
                                                  self.filter_network,
                                                  cutoff_network=self.cutoff_network,
                                                  activation=schnetpack2.nn.activations.shifted_softplus,
                                                  normalize_filter=normalize_filter,
                                                  sdr=sdr, bn=bn,p=p,var_coeff=var_coeff)
        if bn:
            self.dense = schnetpack2.custom.nn.layers.BN_drop_lin(n_atom_basis, n_atom_basis, bn=True, p=p)
        elif not sdr:
            self.dense = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis)
        else:
            self.dense = Linear_sdr(n_atom_basis, n_atom_basis,var_coeff=var_coeff)

        self.attention = attention

        #if attention != 0:
        #    self.gatconv = schnetpack2.custom.nn.attention.GATConv(n_atom_basis, n_atom_basis, heads=torch.abs(attention),
        #                                                          concat=False)
            #self.gatconv = schnetpack2.custom.nn.attention.GATConv(n_atom_basis, n_atom_basis, heads=attention, concat=False, negative_slope=0.2, dropout=p, bias=True)

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
        #if self.attention < 0:
        #    x = self.gatconv(x)

        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)

        #if self.attention > 0:
        #    v = self.gatconv(v)

        v = self.dense(v)
        return v


class SchNetCutoffInteraction(nn.Module):
    """
    SchNet interaction block for modeling quantum interactions of atomistic systems with cosine cutoff.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_basis, n_spatial_basis, n_filters, cutoff,
                 normalize_filter=False):
        super(SchNetCutoffInteraction, self).__init__()

        # initialize filters
        self.filter_network = nn.Sequential(
            schnetpack2.nn.base.Dense(n_spatial_basis, n_filters,
                                     activation=schnetpack2.nn.activations.shifted_softplus),
            schnetpack2.nn.base.Dense(n_filters, n_filters)
        )

        self.cutoff_network = schnetpack2.custom.nn.CosineCutoff(cutoff)
        # initialize interaction blocks
        self.cfconv = schnetpack2.nn.cfconv.CFConv(n_atom_basis, n_filters, n_atom_basis,
                                                  self.filter_network,
                                                  cutoff_network=self.cutoff_network,
                                                  activation=schnetpack2.nn.activations.shifted_softplus,
                                                  normalize_filter=normalize_filter)
        self.dense = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis)

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
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v = self.dense(v)
        return v


#class SC(nn.Module):
#
#    def __init__(self,n_interactions=1):
#        super(SC, self).__init__()
#        #self.conv = nn.Conv1d(n_interactions+1,1,1)
#        self.linear = nn.Linear(n_interactions+1,1)
#
#    def forward(self,vs):
#        v = torch.cat(vs, -1)
#        B,D1,D2,N = v.shape
#        v = v.view(B*D1*D2,N)
#        v = torch.nn.functional.normalize(v,p=2,dim=1)
#        x = self.linear(v)
#        x = x.view(B,D1,D2,1).squeeze()
#        return x
        
class SC(nn.Module):
     def __init__(self, n_interactions, var_coeff=0.1, sdr=False):
        super(SC, self).__init__()
        if not isinstance(n_interactions, list): n_interactions = [n_interactions]
        self.n_interactions = sum(n_interactions)+1
        if sdr:
            self.linear = Linear_sdr(self.n_interactions, 1, activation=None, var_coeff=var_coeff)
        else:
            self.linear = nn.Linear(self.n_interactions,1)
        self.reset_parameters()
        
     def reset_parameters(self):
        print("parameters are reset correctly")
        self.linear.weight.data = torch.ones_like(self.linear.weight.data)/self.n_interactions

     def forward(self, x):
        x = torch.stack(x, dim=-1)
        x = x/(x.pow(2).sum(-1,keepdim=True)+1e-7).sqrt()
        x = self.linear(x)
        return x.squeeze(dim=3)


def get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=5.0, start =0.0, trainable=True):    
    if filter_network_type=="original":
        return None, False
    elif filter_network_type=='laguerre':
        return LaguerreSmearing(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs), True      
    elif filter_network_type=='chebyshev':
        return ChebyshevSmearing(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs, cutoff=cutoff), True
    elif filter_network_type=='gaussian':
        return GaussianSmearing(start, cutoff, n_gaussians=n_filters, trainable=trainable), True


class SchNet(nn.Module):
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
                 return_intermediate=False, max_z=100, cutoff_network=schnetpack2.nn.cutoff.HardCutoff, trainable_gaussians=False,
                 distance_expansion=None, sc=False, start=1.2, debug=False, filter_network_type="original", n_expansion_coeffs=10, sdr=False, bn=False, p=0, var_coeff=0.1, attention=0, return_stress=False):
        super(SchNet, self).__init__()
        self.return_stress = return_stress
        self.debug = debug

        self.cutoff = cutoff

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # spatial features
        self.distances = schnetpack2.custom.nn.neighbors.AtomDistances(return_vecs=return_stress)
        if distance_expansion is None:
            self.distance_expansion = schnetpack2.nn.acsf.GaussianSmearing(start, cutoff, n_gaussians,
                                                                          trainable=trainable_gaussians)
            #self.distance_expansion = schnetpack2.custom.nn.acsf.HIPNNSmearing(1., cutoff, n_gaussians,trainable=True)
        else:
            self.distance_expansion = distance_expansion

        self.return_intermediate = return_intermediate


        # interaction network
        if isinstance(n_interactions,list):
            temp = []
            tempint = 0
            for i in n_interactions:
                filter_network, self.skip_distance_expansion = get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=cutoff, start = start, trainable=trainable_gaussians)
                temp += i*[SchNetInteraction(n_atom_basis=n_atom_basis,
                                  n_spatial_basis=n_gaussians,
                                  n_filters=n_filters,
                                  normalize_filter=normalize_filter, cutoff=self.cutoff,
                                  cutoff_network = cutoff_network,
                                  sdr=sdr, bn=bn, p=p, var_coeff=var_coeff, filter_network=filter_network, attention=attention)]
                tempint += i
            n_interactions = tempint
            self.interactions = nn.ModuleList(temp)
        elif coupled_interactions:
            filter_network, self.skip_distance_expansion = get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=cutoff, start = start, trainable=trainable_gaussians)
            self.interactions = nn.ModuleList([
                                                  SchNetInteraction(n_atom_basis=n_atom_basis,
                                                                    n_spatial_basis=n_gaussians,
                                                                    n_filters=n_filters,
                                                                    normalize_filter=normalize_filter,cutoff=self.cutoff,
                                                                    cutoff_network = cutoff_network,filter_network=filter_network,
                                                                    sdr=sdr, bn=bn, p=p, var_coeff=var_coeff, attention=attention)
                                              ] * n_interactions)
        else:
            interaction_block_list = []
            for _ in range(n_interactions):
                filter_network, self.skip_distance_expansion = get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=cutoff, start = start, trainable=trainable_gaussians)
                interaction_block_list.append(SchNetInteraction(n_atom_basis=n_atom_basis, n_spatial_basis=n_gaussians,
                                  n_filters=n_filters, normalize_filter=normalize_filter,cutoff=self.cutoff,cutoff_network = cutoff_network, sdr=sdr, bn=bn, p=p, var_coeff=var_coeff, filter_network=filter_network, attention=attention))
            
            self.interactions = nn.ModuleList(interaction_block_list)

        self.sc = sc
        
        if self.sc:
            self.sclayer = SC(n_interactions=n_interactions, sdr=sdr, var_coeff=var_coeff)

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
            
        if self.skip_distance_expansion:
            f_ij = None
        elif Structure.f_ij not in inputs.keys():
            f_ij = self.distance_expansion(r_ij)
        else:
            f_ij = inputs[Structure.f_ij]

        # interactions
        if self.return_intermediate:
            xs = [x]

        if self.sc:
            vs = [x]
            #vs = [x.unsqueeze(-1)]

        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v

            if self.return_intermediate:
                xs.append(x)

            if self.sc:
                vs.append(v)
                #vs.append(v.unsqueeze(-1))

        if self.return_intermediate:
            return x, xs

        if self.sc:
            if self.return_stress:
                return self.sclayer(vs), dist_vec
            else:
                return self.sclayer(vs)

        if self.return_stress:
            return x, dist_vec

        return x
