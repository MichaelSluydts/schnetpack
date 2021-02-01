#!/usr/bin/env python3
import torch
from torch.autograd import Variable
from gpytorch.kernels.kernel import Kernel
from gpytorch.functions import RBFCovariance
#from schnetpack2.custom.GP import Hungarian
from scipy.optimize import linear_sum_assignment

import pdb

def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()
    
def cdist_torch(vectors, vectors2):
    """ computes the coupled distances
    Args:
        vectors  is a N x D matrix
        vectors2 is a M x D matrix
        
        returns a N x M matrix
    """
#    pdb.set_trace()
    #distance_matrix = -2 * vectors.mm(torch.t(vectors2)) + vectors.pow(2).sum(dim=1).view(-1, 1) + vectors2.pow(2).sum(dim=1).view(1, -1)
    distance_matrix = (vectors[None,...]-vectors2[:,None,:]).pow(2).sum(-1)
    return distance_matrix#.abs()
    
def material_distances(vectors1, vectors2):
    """ computes the coupled distances
    Args:
        vectors  is a B x N x d matrix
        vectors2 is a D x M x d matrix
        
        returns a B x D matrix
    """
    
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
            
    return similarities


class RBFKernelMaterial(Kernel):
    r"""
    Computes a covariance matrix based on the RBF (squared exponential) kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    .. math::
       \begin{equation*}
          k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left( -\frac{1}{2}
          (\mathbf{x_1} - \mathbf{x_2})^\top \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2}) \right)
       \end{equation*}
    where :math:`\Theta` is a :attr:`lengthscale` parameter.
    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.
    .. note::
        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.
    Args:
        :attr:`ard_num_dims` (int, optional):
            Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if :attr:`x1` is a `n x d` matrix. Default: `None`
        :attr:`batch_shape` (torch.Size, optional):
            Set this if you want a separate lengthscale for each
            batch of input data. It should be `b` if :attr:`x1` is a `b x n x d` tensor. Default: `torch.Size([])`.
        :attr:`active_dims` (tuple of ints, optional):
            Set this if you want to compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions. Default: `None`.
        :attr:`lengthscale_prior` (Prior, optional):
            Set this if you want to apply a prior to the lengthscale parameter.  Default: `None`.
        :attr:`lengthscale_constraint` (Constraint, optional):
            Set this if you want to apply a constraint to the lengthscale parameter. Default: `Positive`.
        :attr:`eps` (float):
            The minimum value that the lengthscale can take (prevents divide by zero errors). Default: `1e-6`.
    Attributes:
        :attr:`lengthscale` (Tensor):
            The lengthscale parameter. Size/shape of parameter depends on the
            :attr:`ard_num_dims` and :attr:`batch_shape` arguments.
    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Non-batch: ARD (different lengthscale for each input dimension)
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=5))
        >>> covar = covar_module(x)  # Output: LazyTensor of size (10 x 10)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])))
        >>> covar = covar_module(x)  # Output: LazyTensor of size (2 x 10 x 10)
    """

    def __init__(self, **kwargs):
        super(RBFKernelMaterial, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return material_distances(x1_, x2_)
                                   
#        return RBFCovariance().apply(x1, x2, self.lengthscale,
#                                     lambda x1, x2: self.covar_dist(x1, x2,
#                                                                    square_dist=True,
#                                                                    diag=False,
#                                                                    dist_postprocess_func=postprocess_rbf,
#                                                                    postprocess=False,
#                                                                    **params))