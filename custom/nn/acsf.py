import torch
from torch import nn as nn

import pdb

from schnetpack2.nn.cutoff import CosineCutoff

__all__ = [
    'AngularDistribution', 'RadialDistribution', 'GaussianSmearing', 'LaguerreSmearing', 'ChebyshevSmearing', 'HIPNNSmearing'
]

def gaussian_smearing(distances, offset, widths, centered=False):
    """
    Perform gaussian smearing on interatomic distances.

    Args:
        distances (torch.Tensor): Variable holding the interatomic distances (B x N_at x N_nbh)
        offset (torch.Tensor): torch tensor of offsets
        centered (bool): If this flag is chosen, Gaussians are centered at the origin and the
                  offsets are used to provide their widths (used e.g. for angular functions).
                  Default is False.

    Returns:
        torch.Tensor: smeared distances (B x N_at x N_nbh x N_gauss)

    """
    if centered == False:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components
        diff = distances[:, :, :, None] - offset[None, None, None, :]
    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances[:, :, :, None]
    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self, start=0.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        return gaussian_smearing(distances, self.offsets, self.width, centered=self.centered)


def hipnn_smearing(distances, offset, widths, centered=False):
    """
    Perform gaussian smearing on interatomic distances.

    Args:
        distances (torch.Tensor): Variable holding the interatomic distances (B x N_at x N_nbh)
        offset (torch.Tensor): torch tensor of offsets
        centered (bool): If this flag is chosen, Gaussians are centered at the origin and the
                  offsets are used to provide their widths (used e.g. for angular functions).
                  Default is False.

    Returns:
        torch.Tensor: smeared distances (B x N_at x N_nbh x N_gauss)

    """
    if centered == False:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 * torch.pow(widths, 2)
        # Use advanced indexing to compute the individual components

        diff = torch.pow(distances[:, :, :, None]+1e-4,-1) - offset[None, None, None, :]
    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 * torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances[:, :, :, None]
    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))
    return gauss


class HIPNNSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self, start=1.0, stop=5.0, n_gaussians=50, centered=False, trainable=False):
        super(HIPNNSmearing, self).__init__()
        offset = torch.linspace(1/stop, 1/start, n_gaussians)
        widths = torch.FloatTensor( 0.2*torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
            #pdb.set_trace()
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        #pdb.set_trace()
        return hipnn_smearing(distances, self.offsets, self.width, centered=self.centered)



def recursive_laguerre(distances, n_expansion_coeffs):
    expansion = []
    expansion.append(1-distances)
    expansion.append((distances.pow(2)-4*distances+2)/2)

    for k in range(2,n_expansion_coeffs):
        expansion.append(((2*k+1-distances)*expansion[k-1]-k*expansion[k-2])/(k+1))

    return torch.stack(expansion, dim=-1)

def recursive_chebyshev(distances, n_expansion_coeffs):
    expansion = []
    expansion.append(distances)
    expansion.append(2*distances*expansion[0]-1)

    for k in range(2,n_expansion_coeffs):
        expansion.append(2*distances*expansion[k-1]-expansion[k-2])

    return torch.stack(expansion, dim=-1)

class PolynomialExpansion(nn.Module):
    """
    Places a predefined number of Laguerre functions within the specified limits.

    Args:
        n_filters (int): Total number of filters.
        n_expansion_coeffs (int): Total number of expansion coefficients of Laguerre Polynomials
        trainable (bool): If set to True, the lengthscales of laguere polynomials are adjusted during training. Default
              is False.
    """

    def __init__(self, n_filters=16, n_expansion_coeffs = 16):
        super(PolynomialExpansion, self).__init__()
        self.n_expansion_coeffs     = n_expansion_coeffs
        self.linear_combs           = nn.Linear(n_expansion_coeffs, n_filters, bias=False)
        self.weights_normalization  = nn.utils.weight_norm(self.linear_combs, name='weight')

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of expanded distances.

        """
        expansions = self.get_expansion(distances)
        return self.weights_normalization(expansions)

    def get_expansion(self, distances):
        raise NotImplementedError

class LaguerreSmearing(PolynomialExpansion):

    def __init__(self, n_filters=16, n_expansion_coeffs = 16):
        super(LaguerreSmearing, self).__init__(n_filters=n_filters, n_expansion_coeffs = n_expansion_coeffs)

    def get_expansion(self, distances):
        return recursive_laguerre(distances, self.n_expansion_coeffs)#recursive_laguerre(4*distances, self.n_expansion_coeffs)*torch.exp(-2*distances).unsqueeze(-1).repeat(1,1,1,self.n_expansion_coeffs)#distances times 4 to correct with reduced bohr radius

class ChebyshevSmearing(PolynomialExpansion):
    """
    Places a predefined number of Laguerre functions within the specified limits.

    Args:
        cutoff:float indicating the size of the atom environment
    """

    def __init__(self, n_filters=16, n_expansion_coeffs = 16, cutoff=5.0):
        super(ChebyshevSmearing, self).__init__(n_filters = n_filters, n_expansion_coeffs = n_expansion_coeffs)
        self.cutoff = cutoff

    def forward(self, distances):
        expansions = self.get_expansion(2*distances/self.cutoff-1)
        return self.weights_normalization(expansions)

    def get_expansion(self, distances):
        return recursive_chebyshev(distances, self.n_expansion_coeffs)


class AngularDistribution(nn.Module):
    """
    Routine used to compute angular type symmetry functions between all atoms i-j-k, where i is the central atom.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        angular_filter (callable): Function used to expand angles between triples of atoms (e.g. BehlerAngular)
        cutoff_functions (callable): Cutoff function
        crossterms (bool): Include radial contributions of the distances r_jk
        pairwise_elements (bool): Recombine elemental embedding vectors via an outer product. If e.g. one-hot encoding
            is used for the elements, this is equivalent to standard Behler functions
            (default=False).

    """

    def __init__(self,
                 radial_filter,
                 angular_filter,
                 cutoff_functions=CosineCutoff,
                 crossterms=False,
                 pairwise_elements=False,
                 aggregate =True
                 ):
        super(AngularDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.angular_filter = angular_filter
        self.cutoff_function = cutoff_functions
        self.crossterms = crossterms
        self.pairwise_elements = pairwise_elements
        self.aggregate = aggregate

    def forward(self, r_ij, r_ik, r_jk, triple_masks=None, elemental_weights=None):
        """
        Args:
            r_ij (torch.Tensor): Distances to neighbor j
            r_ik (torch.Tensor): Distances to neighbor k
            r_jk (torch.Tensor): Distances between neighbor j and k
            triple_masks (torch.Tensor): Tensor mask for non-counted pairs (e.g. due to cutoff)
            elemental_weights (tuple of two torch.Tensor): Weighting functions for neighboring elements, first is for
                                                            neighbors j, second for k

        Returns:
            torch.Tensor: Angular distribution functions

        """

        nbatch, natoms, npairs = r_ij.size()

        # compute gaussilizated distances and cutoffs to neighbor atoms
        radial_ij = self.radial_filter(r_ij)
        radial_ik = self.radial_filter(r_ik)
        angular_distribution = radial_ij * radial_ik

        if self.crossterms:
            radial_jk = self.radial_filter(r_jk)
            angular_distribution = angular_distribution * radial_jk

        # Use cosine rule to compute cos( theta_ijk )
        cos_theta = (torch.pow(r_ij, 2) + torch.pow(r_ik, 2) - torch.pow(r_jk, 2)) / (2.0 * r_ij * r_ik)

        # Required in order to catch NaNs during backprop
        if triple_masks is not None:
            cos_theta[triple_masks == 0] = 0.0

        angular_term = self.angular_filter(cos_theta)

        if self.cutoff_function is not None:
            cutoff_ij = self.cutoff_function(r_ij)
            cutoff_ik = self.cutoff_function(r_ik)
            angular_distribution = angular_distribution * cutoff_ij.unsqueeze(-1) * cutoff_ik.unsqueeze(-1)

            if self.crossterms:
                cutoff_jk = self.cutoff_function(r_jk).squeeze(-1)[...,None]
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
            if self.aggregate:
                return torch.einsum('banr,banf->barf', angular_distribution, angular_term).reshape((nbatch, natoms, -1))
            else:
                angular_distribution = angular_distribution[:, :, :, :, None] * angular_term[:, :, :, None, :]

        if not self.aggregate:
            return angular_distribution.view(nbatch, natoms, angular_distribution.size()[2], -1)

        # For the sum over all contributions
        angular_distribution = torch.sum(angular_distribution, 2)
        # Finally, we flatten the last two dimensions
        angular_distribution = angular_distribution.view(nbatch, natoms, -1)

        return angular_distribution

class RadialDistribution(nn.Module):
    """
    Radial distribution function used e.g. to compute Behler type radial symmetry functions.

    Args:
        radial_filter (callable): Function used to expand distances (e.g. Gaussians)
        cutoff_function (callable): Cutoff function
    """

    def __init__(self, radial_filter, cutoff_function=CosineCutoff):
        super(RadialDistribution, self).__init__()
        self.radial_filter = radial_filter
        self.cutoff_function = cutoff_function

    def forward(self, r_ij, elemental_weights=None, neighbor_mask=None):
        """
        Args:
            r_ij (torch.Tensor): Interatomic distances
            elemental_weights (torch.Tensor): Element-specific weights for distance functions
            neighbor_mask (torch.Tensor): Mask to identify positions of neighboring atoms

        Returns:
            torch.Tensor: Nbatch x Natoms x Nfilter tensor containing radial distribution functions.
        """

        nbatch, natoms, nneigh = r_ij.size()

        radial_distribution = self.radial_filter(r_ij)

        # If requested, apply cutoff function
        if self.cutoff_function is not None:
            cutoffs = self.cutoff_function(r_ij).squeeze(-1)[...,None]
            radial_distribution = radial_distribution * cutoffs

        # Apply neighbor mask
        if neighbor_mask is not None:
            radial_distribution = radial_distribution * torch.unsqueeze(neighbor_mask, -1)

        # Weigh elements if requested
        if elemental_weights is not None:
            radial_distribution = radial_distribution[:, :, :, :, None] * elemental_weights[:, :, :, None, :].float()

        radial_distribution = torch.sum(radial_distribution, 2)
        radial_distribution = radial_distribution.view(nbatch, natoms, -1)
        return radial_distribution