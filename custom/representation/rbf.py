import torch.nn as nn

import schnetpack2.nn.acsf
import schnetpack2.nn.activations
import schnetpack2.nn.base
import schnetpack2.nn.cfconv
import schnetpack2.nn.neighbors
from schnetpack2.custom.data import Structure
import pdb


class RBF(nn.Module):

    def __init__(self,cutoff=5.0,distance_expansion=None, n_gaussians=25, trainable_gaussians = False):
        super(RBF, self).__init__()

        # spatial features
        self.distances = schnetpack2.nn.neighbors.AtomDistances()
        if distance_expansion is None:
            self.distance_expansion = schnetpack2.nn.acsf.GaussianSmearing(0.0, cutoff, n_gaussians,
                                                                          trainable=trainable_gaussians)
        else:
            self.distance_expansion = distance_expansion

    def forward(self, inputs):
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

        # spatial features
        if Structure.r_ij not in inputs.keys():
            r_ij = self.distances(positions, neighbors, cell, cell_offset, neighbor_mask=neighbor_mask)
        if Structure.f_ij not in inputs.keys():
            f_ij = self.distance_expansion(r_ij)

        return r_ij, f_ij