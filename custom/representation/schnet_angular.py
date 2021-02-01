import torch.nn as nn
import torch
import schnetpack2.nn.acsf
import schnetpack2.nn.activations
import schnetpack2.nn.base
import schnetpack2.custom.nn.neighbors
import schnetpack2.nn.cutoff
from schnetpack2.custom.data import Structure
from schnetpack2.custom.nn.layers import Linear_sdr
from schnetpack2.custom.nn.acsf import GaussianSmearing, LaguerreSmearing, ChebyshevSmearing, AngularDistribution
import schnetpack2.custom.nn.neighbors
import schnetpack2.custom.nn.cfconv
import pdb
#from torch_geometric.nn import GATConv

class SchNetInteractionAngularAtom(nn.Module):
    """
    SchNet interaction block for modeling quantum interactions of atomistic systems.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_basis, n_spatial_basis, n_filters, n_angular, n_zetas,
                 normalize_filter=False, cutoff=5.0, cutoff_network=schnetpack2.nn.cutoff.HardCutoff, sdr=False, bn=False, filter_network=None, p=0, attention=0):
        super(SchNetInteractionAngularAtom, self).__init__()

        # initialize filters
        if not filter_network:
            if not sdr:
                if not bn:
                    self.filter_network = nn.Sequential(
                        schnetpack2.nn.base.Dense(n_spatial_basis, n_filters,
                                                 activation=schnetpack2.nn.activations.shifted_softplus),
                        schnetpack2.nn.base.Dense(n_filters, n_filters)
                    )
                else:
                    self.filter_network = nn.Sequential(
                        schnetpack2.custom.nn.layers.BN_drop_lin(n_spatial_basis, n_filters,
                                                 actn=schnetpack2.nn.activations.shifted_softplus),
                        schnetpack2.custom.nn.layers.BN_drop_lin(n_filters, n_filters, actn=None)
                    )

            else:
                self.filter_network = nn.Sequential(
                    Linear_sdr(n_spatial_basis, n_filters,
                                             activation=schnetpack2.nn.activations.shifted_softplus),
                    Linear_sdr(n_filters, n_filters)
                )
        else:
            self.filter_network = filter_network

        angular_filter = nn.Sequential(
                        schnetpack2.nn.base.Dense(2*n_angular*n_zetas*n_zetas*2, n_filters,
                                                 activation=schnetpack2.nn.activations.shifted_softplus),
                        schnetpack2.nn.base.Dense(n_filters, n_filters)
                    )

        self.cutoff_network = cutoff_network(cutoff)

        # initialize interaction blocks
        self.cfconv = schnetpack2.custom.nn.cfconv.CFConv(n_atom_basis, n_filters, n_atom_basis,
                                                  self.filter_network,
                                                  cutoff_network=self.cutoff_network,
                                                  activation=None,
                                                  normalize_filter=normalize_filter,
                                                  sdr=sdr, bn=bn,p=p)

        self.cfconv_angular = schnetpack2.custom.nn.cfconv.CFConv(n_atom_basis, n_filters, n_atom_basis,
                                                  angular_filter,
                                                  cutoff_network=self.cutoff_network,
                                                  activation=None,
                                                  normalize_filter=normalize_filter,
                                                  sdr=sdr, bn=bn,p=p)

#        self.cfconv_angular = schnetpack2.custom.nn.cfconv.CFConvAngular(n_atom_basis, n_filters, n_atom_basis,
#                                                  angular_filter,
#                                                  cutoff_network=self.cutoff_network,
#                                                  activation=None,
#                                                  normalize_filter=normalize_filter,
#                                                  sdr=sdr, bn=bn,p=p)

        if bn:
            self.dense = schnetpack2.custom.nn.layers.BN_drop_lin(n_atom_basis, n_atom_basis, bn=True, p=p)
        elif not sdr:
            self.dense = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, activation=None)
            self.dense_rad = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, activation=None)
        else:
            self.dense = Linear_sdr(n_filters, n_filters)
        self.attention = attention
        #if attention > 0:
            #self.gatconv = GATConv(n_atom_basis, n_atom_basis, heads=1, concat=True, negative_slope=0.2, dropout=p, bias=True)

        self.dense_atom = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, bias=False, activation=None)
        self.dense = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, activation=None)
        self.dense_angular = schnetpack2.nn.base.Dense(2*n_zetas*2*n_angular*n_zetas, n_atom_basis, bias=False, activation=None)
        self.dense_tot = schnetpack2.nn.base.Dense(n_atom_basis, n_atom_basis, activation=None)

    def forward(self, x, r_ij, neighbors, neighbor_mask, neighbors_i, neighbors_k, neighbor_mask_triples, G_i, f_ij=None):
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
        v_rad  = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij)
        v_rad  = self.dense(v_rad)
#        v = self.cfconv_angular(
#        v  = self.dense_atom(x)
#        v += self.dense_angular(G_i)
        #v_ang = self.cfconv_angular(x, G_i, neighbors_i, neighbors_k, neighbor_mask_triples, None)
        #v_rad  = self.cfconv(x, r_ij, neighbors, neighbor_mask, None)

        v_ang  = self.dense_angular(G_i)
#        v = torch.cat([x, v_rad, v_ang], dim=-1)
        v = v_rad + v_ang
#        v = self.dense_tot(v)
        return schnetpack2.nn.activations.shifted_softplus(v)

#        return self.ADF(r_ij, r_ik, r_jk, elemental_weights=(Z_ij, Z_ik), triple_masks=neighbor_pairs_mask)

class SC(nn.Module):
     def __init__(self, n_interactions):
        super(SC, self).__init__()
        if not isinstance(n_interactions, list): n_interactions = [n_interactions]
        self.n_interactions = sum(n_interactions)+1
        self.linear = nn.Linear(self.n_interactions,1)
        self.reset_parameters()

     def reset_parameters(self):
        print("parameters are reset correctly")
        self.linear.weight.data = torch.ones_like(self.linear.weight.data)/self.n_interactions

     def forward(self, x):
        x = torch.stack(x, dim=-1)
        x = x/(x.pow(2).sum(-1,keepdim=True)+1e-7).sqrt()
        x = self.linear(x)
        return x.squeeze()


def get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=5.0, start =0.0, trainable=True):
    if filter_network_type=="original":
        return None, False
    elif filter_network_type=='laguerre':
        return LaguerreSmearing(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs), True
    elif filter_network_type=='chebyshev':
        return ChebyshevSmearing(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs, cutoff=cutoff), True
    elif filter_network_type=='gaussian':
        return GaussianSmearing(start, cutoff, n_gaussians=n_filters, trainable=trainable), True

class SchNetAngular(nn.Module):
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

    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=1, n_angular=8, cutoff=5.0, n_gaussians=25,
                 zetas={1, 2, 3, 4},normalize_filter=False, coupled_interactions=False,
                 return_intermediate=False, max_z=100, cutoff_network=schnetpack2.nn.cutoff.HardCutoff, trainable_gaussians=False,
                 distance_expansion=None, sc=False, start=1.2, debug=False, filter_network_type="original", n_expansion_coeffs=10, sdr=False, bn=False, p=0, attention=0, return_stress=False):
        super(SchNetAngular, self).__init__()
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
        else:
            self.distance_expansion = distance_expansion

        self.return_intermediate = return_intermediate


        # interaction network
        if isinstance(n_interactions,list):
            temp = []
            tempint = 0
            for i in n_interactions:
                filter_network, self.skip_distance_expansion = get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=cutoff, start = start, trainable=trainable_gaussians)
                temp += i*[SchNetInteractionAngularAtom(n_atom_basis=n_atom_basis,
                                  n_spatial_basis=n_gaussians,
                                  n_filters=n_filters,
                                  n_angular = n_angular,
                                  n_zetas = len(zetas),
                                  normalize_filter=normalize_filter, cutoff=self.cutoff,
                                  cutoff_network = cutoff_network,
                                  sdr=sdr, bn=bn, p=p, filter_network=filter_network, attention=attention)]
                tempint += i
            n_interactions = tempint
            self.interactions = nn.ModuleList(temp)
        elif coupled_interactions:
            filter_network, self.skip_distance_expansion = get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=cutoff, start = start, trainable=trainable_gaussians)
            self.interactions = nn.ModuleList([
                                               SchNetInteractionAngularAtom(n_atom_basis=n_atom_basis,
                                                                    n_spatial_basis=n_gaussians,
                                                                    n_filters=n_filters,
                                                                    n_angular = n_angular,
                                                                    n_zetas = len(zetas),
                                                                    normalize_filter=normalize_filter,cutoff=self.cutoff,
                                                                    cutoff_network = cutoff_network,filter_network=filter_network,
                                                                    sdr=sdr, bn=bn, p=p, attention=attention)
                                              ] * n_interactions)
        else:
            interaction_block_list = []
            for _ in range(n_interactions):
                filter_network, self.skip_distance_expansion = get_filter_network(filter_network_type, n_filters, n_expansion_coeffs, cutoff=cutoff, start = start, trainable=trainable_gaussians)
                interaction_block_list.append(SchNetInteractionAngularAtom(n_atom_basis=n_atom_basis, n_spatial_basis=n_gaussians, n_filters=n_filters, n_angular = n_angular, n_zetas = len(zetas), normalize_filter=normalize_filter,cutoff=self.cutoff,cutoff_network = cutoff_network, sdr=sdr, bn=bn, p=p, filter_network=filter_network, attention=attention))

            self.interactions = nn.ModuleList(interaction_block_list)

        self.sc = sc

        if self.sc:
            self.sclayer = SC(n_interactions=n_interactions)

        angular_filter = schnetpack2.nn.acsf.BehlerAngular(zetas=zetas)
        radial_filter =  schnetpack2.nn.acsf.GaussianSmearing(start=1.0, stop = cutoff - 0.5, n_gaussians=2*n_angular*len(zetas), centered=True)
        self.ADF =  AngularDistribution(radial_filter, angular_filter, cutoff_functions=cutoff_network(cutoff),
                                                   crossterms=False, pairwise_elements=False, aggregate=True)

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
        cell_offset_j = inputs[Structure.cell_offset_j]
        cell_offset_k = inputs[Structure.cell_offset_k]
        neighbors = inputs[Structure.neighbors]
        neighbor_mask = inputs[Structure.neighbor_mask]
        idx_j = inputs[Structure.neighbor_pairs_j]
        idx_k = inputs[Structure.neighbor_pairs_k]
        neighbor_pairs_mask = inputs[Structure.neighbor_pairs_mask]

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

#        G_i = self.compute_angular_contribution(x, positions, neighbor_mask, idx_j, idx_k, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k)
        G_i = self.ADF(*self.compute_triangle_sides(x, positions, neighbor_mask, idx_j, idx_k, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k),triple_masks=neighbor_pairs_mask)

        # interactions
        if self.return_intermediate:
            xs = [x]

        if self.sc:
            vs = [x.unsqueeze(-1)]

        for interaction in self.interactions:
            v = interaction(x, r_ij, neighbors, neighbor_mask, idx_j, idx_k, neighbor_pairs_mask, G_i, f_ij=f_ij)
            x = x + v

            if self.return_intermediate:
                xs.append(x)

            if self.sc:
                vs.append(v.unsqueeze(-1))

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

    def compute_angular_contribution(self, x, positions, neighbor_mask, idx_j, idx_k, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k):
#        Z_ij  = schnetpack2.nn.neighbors.neighbor_elements(x, idx_j)
#        Z_ik  = schnetpack2.nn.neighbors.neighbor_elements(x, idx_k)
        r_ij, r_ik, r_jk = schnetpack2.nn.custom.neighbors.triple_distances_masked(positions, idx_j, idx_k, neighbor_mask, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k)
        cos_theta_jik = (r_ik.pow(2) + r_ij.pow(2) -r_jk.pow(2))/(2.0*r_ik*r_ij)#self.ADF(r_ij, r_ik, r_jk, elemental_weights=None, triple_masks=neighbor_pairs_mask)
        if neighbor_pairs_mask is not None:
            cos_theta_jik[neighbor_pairs_mask == 0] = 0.0
        if self.cutoff_function is not None:
            cutoff_ij = self.cutoff_function(r_ij)
            cutoff_ik = self.cutoff_function(r_ik)
            angular_distribution = angular_distribution * cutoff_ij.unsqueeze(-1) * cutoff_ik.unsqueeze(-1)

            if self.crossterms:
                cutoff_jk = self.cutoff_function(r_jk)
                angular_distribution = angular_distribution * cutoff_jk

        # Compute radial part of descriptor
        if triple_masks is not None:
            # Filter out nan divisions via boolean mask, since
            # angular_term = angular_term * triple_masks
            # is not working (nan*0 = nan)
            angular_term[triple_masks == 0] = 0.0
            triple_masks = torch.unsqueeze(triple_masks, -1)
            angular_distribution = angular_distribution * triple_masks

        # Apply weights here, since dimension is still the same
        if elemental_weights is not None:
            if not self.pairwise_elements:
                Z_ij, Z_ik = elemental_weights
                Z_ijk = Z_ij * Z_ik
                angular_distribution = torch.unsqueeze(angular_distribution, -1) * torch.unsqueeze(Z_ijk, -2).float()
            else:
                # Outer product to emulate vanilla SF behavior
                Z_ij, Z_ik = elemental_weights
                B, A, N, E = Z_ij.size()
                pair_elements = Z_ij[:, :, :, :, None] * Z_ik[:, :, :, None, :]
                pair_elements = pair_elements + pair_elements.permute(0, 1, 2, 4, 3)
                # Filter out lower triangular components
                pair_filter = torch.triu(torch.ones(E, E)) == 1
                pair_elements = pair_elements[:, :, :, pair_filter]
                angular_distribution = torch.unsqueeze(angular_distribution, -1) * torch.unsqueeze(pair_elements, -2)

            # Dimension is (Nb x Nat x Nneighpair x Nrad) for angular_distribution and
            # (Nb x Nat x NNeigpair x Nang) for angular_term, where the latter dims are orthogonal
            # To multiply them:
            angular_distribution = angular_distribution[:, :, :, :, None, :] * angular_term[:, :, :, None, :, None]
        else:
            angular_distribution = angular_distribution[:, :, :, :, None] * angular_term[:, :, :, None, :]
        # For the sum over all contributions
        angular_distribution = torch.sum(angular_distribution, 2)
        return cos_theta_jik

    def compute_triangle_sides(self, x, positions, neighbor_mask, idx_j, idx_k, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k):
#        Z_ij  = schnetpack2.nn.neighbors.neighbor_elements(x, idx_j)
#        Z_ik  = schnetpack2.nn.neighbors.neighbor_elements(x, idx_k)
#        r_ij, r_ik, r_jk = schnetpack2.nn.neighbors.triple_distances_masked(positions, idx_j, idx_k, neighbor_mask, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k)
        return schnetpack2.custom.nn.neighbors.triple_distances_masked(positions, idx_j, idx_k, neighbor_mask, neighbor_pairs_mask, cell, cell_offset_j, cell_offset_k)