import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy

import torch
import torch.nn as nn

import pdb

class GPRegressionLayer(AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, inputs, **kwargs):
        return self.variational_strategy(inputs)

class DKLModel(gpytorch.Module):
    def __init__(self, inducing_points):
        super(DKLModel, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=inducing_points.shape[-1])
        inducing_points = (inducing_points-inducing_points.mean(0, keepdim=True))/inducing_points.std(0, keepdim=True)
        inducing_points += (torch.zeros_like(inducing_points).uniform_()-1/2)*0.1
        self.gp_layer = GPRegressionLayer(inducing_points)

    def forward(self, inputs):
    
        if inputs['representation'].dim() == 1:
            inputs = inputs['representation'].unsqueeze(-1)  

        if inputs['representation'].dim() == 3:
            B, A, D = inputs['representation'].shape
            inputs = inputs['representation'].view(B*A, D)

        inputs = self.bn(inputs)

        res = self.gp_layer(inputs)
        
        return res
        
class FlattenedMLL(nn.Module):
    def __init__(self, gp_model, num_data, kf=0.0):
        super(FlattenedMLL, self).__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, gp_model, num_data=num_data, combine_terms=True)
        self.kf = kf

    def forward(self, inputs, targets):
        mll_E = self.mll(inputs["y_dist"], targets["energy"]).mean()
        fdiff  = targets['forces'].type(inputs["y"].dtype) - inputs['dydx']
        fdiff  = fdiff.pow(2)
        N= 3*torch.sum((inputs['natoms'] != 0).type(inputs["y"].dtype), 1, keepdim=True)
        fmean  = torch.mean(torch.sum(fdiff,[1,2])/N.t()/3)
        return mll_E.mean() + self.kf*fmean