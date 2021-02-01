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

class Parameter_dict(Parameter):
#    def __init__(self, dict_of_parameters):
#        print(dict_of_parameters.keys())
#        if "weight" in dict_of_parameters:
#          super( Parameter_dict["weigth"], self ).__init__()
#        if "bias" in dict_of_parameters:
#          super( Parameter_dict["bias"], self ).__init__() 
#        self.dict = dict_parameters

    def __new__(cls, parameter, dict_of_parameters, requires_grad = True):
#        pdb.set_trace()
#        if "weight" in dict_of_parameters:
        obj = Parameter.__new__(cls, parameter, requires_grad)
#          del dict_of_parameters["weight"]
#        if "bias" in dict_of_parameters:
#          obj = Parameter.__new__(cls, dict_of_parameters["bias"].data, requires_grad)
#          del dict_of_parameters["bias"]
        obj.dict = dict_of_parameters
        return obj
#        self.required_grad = any(self.dict[key].requires_grad for key in self.dict)
#        self.volatile      = any(self.dict[key].volatile      for key in self.dict)
#        self.is_leaf       = any(self.dict[key].is_leaf       for key in self.dict)
    def cuda(self):
        print("function has been called")
        return Parameter_dict(super(self).cuda(), {k:v.cuda() for k,v in self.dct})

    def attach(self):
        print("function has been called")
        return Parameter_dict(super(self).attach(), self.dct)
        
    def to(self, device):
        print("function has been called")
        return Parameter_dict(super(self).to(device), {k:v.to(device) for k,v in self.dct})
        

class shifted_softplus(nn.Module):
    """
    Shifted softplus activation function of the form:
    :math:`y = ln( e^{-x} + 1 ) - ln(2)`

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Shifted softplus applied to x

    """

    def __init__(self):
        super(shifted_softplus,self).__init__()
    def forward(self,x):
        return F.softplus(x) - np.log(2.0)

class Linear_sdr(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None, var_coeff = 0.1, p=0.0):
        super(Linear_sdr, self).__init__()
        
        self.var_coeff = var_coeff
        self.p = p
        
        self.in_features = in_features
        self.out_features = out_features
            
        if cuda:
            self.noise_weight = torch.FloatTensor(1,1).cuda()
            self.noise_bias = torch.FloatTensor(1).cuda()
            self.register_buffer('means_weight', torch.Tensor(out_features, in_features).cuda())
            self.register_buffer('stds_weight', torch.Tensor(out_features, in_features).cuda())
            if bias:
                self.register_buffer('means_bias', torch.Tensor(out_features).cuda())
                self.register_buffer('stds_bias', torch.Tensor(out_features).cuda())
            else:
                self.register_parameter('bias', None)
        else:
            self.noise_weight = torch.FloatTensor(1,1)
            self.noise_bias   = torch.FloatTensor(1)
            self.register_buffer(means_weight, torch.Tensor(out_features, in_features))
            self.register_buffer(stds_weight, torch.Tensor(out_features, in_features))
            if bias:
                self.register_buffer('means_bias', torch.Tensor(out_features))
                self.register_buffer('stds_bias', torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
#        self.weight_mean = Parameter(torch.tensor([-2.5]))
#        self.weight_std  = Parameter(torch.tensor([1.0]))
#        self.weight_mean = torch.tensor([-2.5]).cuda()
#        self.weight_std  = torch.tensor([1.0]).cuda()
        
        self.weight = Parameter_dict(torch.Tensor(out_features, in_features), {'means_weight' : self.means_weight, 'stds_weight' : self.stds_weight, 'var_coeff':var_coeff})
        if bias:
#            self.bias_var_coeff = torch.tensor([var_coeff]).cuda()#Parameter(torch.tensor([var_coeff]))
            self.bias   = Parameter_dict(torch.Tensor(out_features), {'means_bias' : self.means_bias, 'stds_bias' : self.stds_bias, 'var_coeff':var_coeff})
            #self.bias   = Parameter_dict(torch.Tensor(out_features), {'means_bias' : self.means_bias, 'stds_bias' : self.stds_bias, "var_coeff":self.bias_var_coeff, "noise":self.noise_bias})
#            self.bias_mean = torch.tensor([-2.0]).cuda()
#            self.bias_std  = torch.tensor([0.5]).cuda()
                 
        self.activation = activation
        
        self.dropout = nn.Dropout(p)
        
#        self.noise_layer = GaussianNoise(0.05)
                                      
        self.reset_parameters()

    def reset_parameters(self):
#        pdb.set_trace()
        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
        self.weight.dict['means_weight'].uniform_(-stdv, stdv)
        self.weight.dict['stds_weight'].uniform_(10e-8, stdv/10)
        
        if self.bias is not None:
            self.bias.dict['means_bias'].uniform_(-stdv, stdv)
            self.bias.dict['stds_bias'].uniform_(10e-8, stdv/10)
        
    def forward(self, input):
#        if self.training:
#        print(self.weight.dict['means_weight'].abs().max())
#        print(self.weight.dict['stds_weight'].abs().max())

#        if not self.training:
#            if self.bias is not None:
#                output = F.linear(input, self.weight.dict['means_weight'], self.bias.dict['means_bias'])
#            else:
#                output = F.linear(input, self.weight.dict['means_weight'])
#            
#            if self.activation is not None:
#                return self.activation(output)
#            else:
#                return output
        
#        input = self.noise_layer(input)
#        self.weight.dict["var_coeff"] = self.weight_var_coeff*self.noise_weight.repeat(*self.weight.data.size()).exponential_()    
#        self.weight.data = self.noise_weight.repeat(*self.weight.data.size()).normal_()*self.weight.dict['stds_weight']+self.weight.dict['means_weight']
#        self.weight_var_coeff.data = self.noise_weight.repeat(*self.weight.data.size()).normal_()

        #stds = self.weight_var_coeff*self.weight.dict['means_weight']#self.noise_weight.repeat(*self.weight.data.size()).exponential_()*self.weight_var_coeff*self.weight.dict['means_weight']
        #stds = self.weight.dict['means_weight']*self.noise_weight.repeat(*self.weight.data.size()).log_normal_(mean=-2.5, std=1)
        #stds = self.weight.dict['means_weight']*torch.exp(self.noise_weight.repeat(*self.weight.data.size()).normal_()*self.weight_std+self.weight_mean)
#        stds = self.weight.dict['means_weight']*torch.exp(self.noise_weight.repeat(*self.weight.data.size()).normal_()*self.weight_std+self.weight_mean)
        self.weight.data = self.noise_weight.repeat(*self.weight.data.size()).normal_()*self.weight.dict['stds_weight']+self.weight.dict['means_weight']

        if self.bias is not None:
#          self.bias.dict["var_coeff"] = self.bias_var_coeff*self.noise_bias.repeat(*self.bias.data.size()).exponential_()
#          self.bias.data   = self.noise_bias.repeat(*self.bias.data.size()).normal_()*self.bias.dict['stds_bias']+self.bias.dict['means_bias']
          #stds = self.bias_var_coeff*self.bias.dict['means_bias']#self.noise_bias.repeat(*self.bias.data.size()).exponential_()*self.bias_var_coeff*self.bias.dict['means_bias']   
          #stds = self.bias.dict['means_bias']*self.noise_bias.repeat(*self.bias.data.size()).log_normal_(mean=-2.5, std=1)
#          stds = self.bias.dict['means_bias']*torch.exp(self.noise_bias.repeat(*self.bias.data.size()).normal_()*self.bias_std+self.bias_mean)
          self.bias.data = self.noise_bias.repeat(*self.bias.data.size()).normal_()*self.bias.dict['stds_bias']+self.bias.dict['means_bias']
          
        if self.activation is not None:
          return self.dropout(self.activation(F.linear(input, self.weight, self.bias)))
        else:
          return self.dropout(F.linear(input, self.weight, self.bias))
          
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', p=' + str(self.p)\
            + ', var coeff=' + str(self.var_coeff)\
            + ', bias=' + str(self.bias is not None) + ')'
            
class Linear_sdr_Reversed(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None, var_coeff = 0.1, p=0.0):
        super(Linear_sdr_Reversed, self).__init__()
        
        self.var_coeff = var_coeff
        self.p = p
        
        self.in_features = in_features
        self.out_features = out_features
            
        if cuda:
            self.noise_weight = torch.FloatTensor(1,1).cuda()
            self.noise_bias = torch.FloatTensor(1).cuda()
            self.register_buffer('means_weight', torch.Tensor(out_features, in_features).cuda())
            self.register_buffer('stds_weight', torch.Tensor(out_features, in_features).cuda())
            if bias:
                self.register_buffer('means_bias', torch.Tensor(out_features).cuda())
                self.register_buffer('stds_bias', torch.Tensor(out_features).cuda())
            else:
                self.register_parameter('bias', None)
        else:
            self.noise_weight = torch.FloatTensor(1,1)
            self.noise_bias   = torch.FloatTensor(1)
            self.register_buffer(means_weight, torch.Tensor(out_features, in_features))
            self.register_buffer(stds_weight, torch.Tensor(out_features, in_features))
            if bias:
                self.register_buffer('means_bias', torch.Tensor(out_features))
                self.register_buffer('stds_bias', torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)
        
        self.weight = Parameter_dict(torch.Tensor(out_features, in_features), {'means_weight' : self.means_weight, 'stds_weight' : self.stds_weight, 'var_coeff':var_coeff})
        if bias:
            self.bias   = Parameter_dict(torch.Tensor(out_features), {'means_bias' : self.means_bias, 'stds_bias' : self.stds_bias, 'var_coeff':var_coeff})
                 
        self.activation = activation
        
        self.dropout = nn.Dropout(p)
        
#        self.noise_layer = GaussianNoise(0.05)
                                      
        self.reset_parameters()

    def reset_parameters(self):
#        pdb.set_trace()
        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
        self.weight.dict['means_weight'].uniform_(-stdv, stdv)
        self.weight.dict['stds_weight'].uniform_(10e-8, stdv/10)
        
        if self.bias is not None:
            self.bias.dict['means_bias'].uniform_(-stdv, stdv)
            self.bias.dict['stds_bias'].uniform_(10e-8, stdv/10)
        
    def forward(self, input):
#        if self.training:
#        print(self.weight.dict['means_weight'].abs().max())
#        print(self.weight.dict['stds_weight'].abs().max())

#        if not self.training:
#            if self.bias is not None:
#                output = F.linear(input, self.weight.dict['means_weight'], self.bias.dict['means_bias'])
#            else:
#                output = F.linear(input, self.weight.dict['means_weight'])
#            
#            if self.activation is not None:
#                return self.activation(output)
#            else:
#                return output

        if self.activation is not None:        
            input = self.activation(input)
        
        self.weight.data = self.noise_weight.repeat(*self.weight.data.size()).normal_()*self.weight.dict['stds_weight']+self.weight.dict['means_weight']

        if self.bias is not None:
          self.bias.data = self.noise_bias.repeat(*self.bias.data.size()).normal_()*self.bias.dict['stds_bias']+self.bias.dict['means_bias']
          

        return self.dropout(F.linear(input, self.weight, self.bias))
          
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', p=' + str(self.p)\
            + ', var coeff=' + str(self.var_coeff)\
            + ', bias=' + str(self.bias is not None) + ')'
            
class Linear_sdr_old(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(Linear_sdr, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
            
        if cuda:
            self.noise_weight = torch.FloatTensor(1,1).cuda()
            self.noise_bias = torch.FloatTensor(1).cuda()
            weight_data   = torch.Tensor(out_features, in_features)
            means_weight  = torch.Tensor(out_features, in_features).cuda()
            stds_weight   = torch.Tensor(out_features, in_features).cuda()
            if bias:
                bias_data     = torch.Tensor(out_features)
                means_bias    = torch.Tensor(out_features).cuda()
                stds_bias     = torch.Tensor(out_features).cuda()
            else:
                self.register_parameter('bias', None)
        else:
            self.noise_weight = torch.FloatTensor(1,1)
            self.noise_bias   = torch.FloatTensor(1)
            weight_data   = torch.Tensor(out_features, in_features)
            means_weight  = torch.Tensor(out_features, in_features)
            stds_weight   = torch.Tensor(out_features, in_features)
            if bias:
                bias_data     = torch.Tensor(out_features)
                means_bias    = torch.Tensor(out_features)
                stds_bias     = torch.Tensor(out_features)
            else:
                self.register_parameter('bias', None)
        
        self.weight = Parameter_dict(weight_data, {'means_weight' : means_weight, 'stds_weight' : stds_weight})
        if bias:
          self.bias   = Parameter_dict(bias_data, {'means_bias' : means_bias, 'stds_bias' : stds_bias})
          
        self.activation = activation
        
#        self.noise_layer = GaussianNoise(0.05)
                                      
        self.reset_parameters()

    def reset_parameters(self):
#        pdb.set_trace()
        stdv = 1. / math.sqrt(self.weight.size(1))
#        self.weight.data.uniform_(-stdv, stdv)
        self.weight.dict['means_weight'].uniform_(-stdv, stdv)
        self.weight.dict['stds_weight'].uniform_(10e-8, stdv)
        
        if self.bias is not None:
            self.bias.dict['means_bias'].uniform_(-stdv, stdv)
            self.bias.dict['stds_bias'].uniform_(10e-8, stdv)
        
    def forward(self, input):
#        if self.training:
#        print(self.weight.dict['means_weight'].abs().mean())
#        print(self.weight.dict['stds_weight'].abs().mean())

#        if not self.training:
#            if self.bias is not None:
#                output = F.linear(input, self.weight.dict['means_weight'], self.bias.dict['means_bias'])
#            else:
#                output = F.linear(input, self.weight.dict['means_weight'])
#            
#            if self.activation is not None:
#                return self.activation(output)
#            else:
#                return output
        
        input = self.noise_layer(input)
            
        self.weight.data = self.noise_weight.repeat(*self.weight.data.size()).normal_()*self.weight.dict['stds_weight']+self.weight.dict['means_weight']
        if self.bias is not None:
          self.bias.data   = self.noise_bias.repeat(*self.bias.data.size()).normal_()*self.bias.dict['stds_bias']+self.bias.dict['means_bias']
        if self.activation is not None:
          return self.activation(F.linear(input, self.weight, self.bias))
        else:
          return F.linear(input, self.weight, self.bias)
          
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class GroupBatchNorm2d(nn.Module):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super(GroupBatchNorm2d,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(1, 1, c_num))
        self.beta = nn.Parameter(torch.zeros(1, 1, c_num))
        self.eps = eps

    def forward(self, x):
        B, A, N, D = x.size()

        x = x.permute(0,3, 2, 1).contiguous().view(B, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(B, D, N, A).permute(0,3,2,1)

        return x * self.gamma + self.beta

class GroupBatchNorm1d(nn.Module):
    def __init__(self, c_num, group_num = 16, eps = 1e-10):
        super(GroupBatchNorm1d,self).__init__()
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.ones(1, 1,c_num))
        self.beta = nn.Parameter(torch.zeros(1, 1,c_num))
        self.eps = eps

    def forward(self, x):
        B, A, D = x.size()

        x = x.permute(0,2, 1).contiguous().view(B, self.group_num, -1)

        mean = x.mean(dim = 2, keepdim = True)
        std = x.std(dim = 2, keepdim = True)

        x = (x - mean) / (std+self.eps)
        x = x.view(B, D, A).permute(0,2,1)

        return x * self.gamma + self.beta


def bn_drop_lin(n_in, n_out, bn:bool=True, p=0., actn=None):
   "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
   layers = [nn.BatchNorm1d(n_in)] if bn else []
#   layers = [GroupBatchNorm1d(n_in)] if bn else []
   if p != 0: layers.append(nn.Dropout(p))
   layers.append(nn.Linear(n_in, n_out))
   if actn is not None: layers.append(actn)
   return layers

class BN_drop_lin(nn.Module):
    def __init__(self, n_in, n_out, bn:bool=True, p=0., actn=schnetpack2.nn.activations.shifted_softplus, bias=True):
        super(BN_drop_lin, self).__init__()
#        self.bn  = nn.GroupNorm(16, n_in)#nn.InstanceNorm1d(n_in)
#        self.bn2 = nn.InstanceNorm2d(n_in)
        self.bn = nn.BatchNorm1d(n_in)
        self.bn2 = nn.BatchNorm2d(n_in)
        self.do = nn.Dropout(p)
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.actn = actn

    def forward(self, x):
        if len(x.shape) == 3:
            #x=self.bn(x)
            B,A,F = x.shape
            x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        elif len(x.shape) == 4:
            #x=self.bn2(x)
            B,A,N,F = x.shape
            x = self.bn2(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.do(x)
        x = self.linear(x)
        if self.actn is not None:
            x = self.actn(x)
        return x
        
class BN_drop_lin_Reversed(nn.Module):
    def __init__(self, n_in, n_out, bn:bool=True, p=0., actn=schnetpack2.nn.activations.shifted_softplus, bias=True):
        super(BN_drop_lin, self).__init__()
#        self.bn  = nn.GroupNorm(16, n_in)#nn.InstanceNorm1d(n_in)
#        self.bn2 = nn.InstanceNorm2d(n_in)
        self.bn = nn.BatchNorm1d(n_in)
        self.bn2 = nn.BatchNorm2d(n_in)
        self.do = nn.Dropout(p)
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.actn = actn

    def forward(self, x):
        if self.actn is not None:
            x = self.actn(x)
        if len(x.shape) == 3:
            #x=self.bn(x)
            B,A,F = x.shape
            x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        elif len(x.shape) == 4:
            #x=self.bn2(x)
            B,A,N,F = x.shape
            x = self.bn2(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.do(x)
        x = self.linear(x)
        return x

def create_output_layer(n_in, n_out, n_layers=2, ps=0.5, bns=True, acts=None, bn_final=True):
   lin_ftrs = [int(n_in*(n_out/n_in)**(i/n_layers)) for i in range(n_layers+1)]
   if not isinstance(ps, list): ps  = [ps]
   if not isinstance(bns, list): bns = [bns]
   if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
   if len(bns) == 1: bns = bns*(len(lin_ftrs)-1)
   if not acts: actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
   elif type(acts) == type(nn.ReLU): actns = [acts()] * (len(lin_ftrs)-2) + [None]
   layers = []
   for ni,no,p,bn,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, bns, actns):
       layers += bn_drop_lin(ni, no, bn, p, actn)
   if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
   return nn.Sequential(*layers)

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        if torch.cuda.is_available():
            self.noise = torch.FloatTensor(1,1).cuda()
        else:
            self.noise = torch.FloatTensor(1,1)
            
    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = Variable(self.noise.repeat(*x.size()).normal_()) * scale
            x = x + sampled_noise
        return x 
  
##  def named_parameters(self, memo=None, prefix=''):
##        pdb.set_trace()
#        together_bias   = dict()
#        together_weight = dict()
#        if memo is None:
#            memo = set()
#        for name, p in self._parameters.items():
#            if p is not None and p not in memo:
#                memo.add(p)
#                if name in ['weight', 'means_weight', 'stds_weight']:
#                  together_weight[name] = p
#                if name in ['bias', 'means_bias', 'stds_bias']:   
#                  together_bias[name]   = p
#        if together_weight:
#            yield prefix + ('.' if prefix else '') + "together", Parameter_dict(together_weight['weight'], {v:k for v,k in together_weight.items() if v != 'weight'})
#        if together_bias:
#            yield prefix + ('.' if prefix else '') + "together",  Parameter_dict(together_bias['bias'], {v:k for v,k in together_bias.items() if v != 'bias'})
#        for mname, module in self.named_children():
#            submodule_prefix = prefix + ('.' if prefix else '') + mname
#            for name, p in module.named_parameters(memo, submodule_prefix):
#                yield name, p

def get_ensemble_stats(preds, sigma, mean, std):
    preds = (preds-mean)/std
    sigma = sigma/std
    mu_ensemble = preds.mean(1)
#    var_ensemble = (sigma.pow(2)+preds.pow(2)).mean(1)-mu_ensemble.pow(2)+1e-6
    var_ensemble = sigma.pow(2).mean(1) + preds.var(1) + 1e-6
    return mu_ensemble*std+mean, var_ensemble.sqrt()*std

#def get_ensemble_stats_forces(preds, sigma):
#    pdb.set_trace()
#    mu_ensemble = preds.mean(-2)
#    var_ensemble = (sigma.pow(2)+preds.pow(2)).mean(-2)-mu_ensemble.pow(2)
#    
#    return mu_ensemble, var_ensemble.sqrt()

class EnsembleModel(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, model, mean, std, mean_forces, std_forces):
        super().__init__()
        self.model = model
        
        self.mean = mean
        self.std  = std
        
        self.mean_forces = mean_forces
        self.std_forces  = std_forces
            
    def forward(self, x, n_samples=10):
        means  = []
        sigmas = []
        means_forces  = []
        sigmas_forces = []
        
        for sample in range(n_samples):
#            torch.cuda.empty_cache()
            result = self.model(x)
            means.append(result['y'].detach().cpu())
            if 'sigma' in result: sigmas.append(result['sigma'].detach().cpu())
            means_forces.append(result['dydx'].detach().cpu())
            if 'sigma_forces' in result: sigmas_forces.append(result['sigma_forces'].detach().cpu())
            del result
            
        means =torch.stack(means,1)
        if len(sigmas)>0: sigmas=torch.stack(sigmas,1)

        means_forces  =torch.stack(means_forces,1)
        if len(sigmas_forces)>0: sigmas_forces =torch.stack(sigmas_forces,1)
        
        results_ensemble = {}
        
#        results_ensemble['y'], results_ensemble['sigma'] = get_ensemble_stats(means, sigmas, self.mean, self.std)
        if len(sigmas)>0: results_ensemble['y'], results_ensemble['sigma'], results_ensemble['sigma_ensemble'] = means.mean(), sigmas.pow(2).mean().sqrt(), means.std()
        else: results_ensemble['y'], results_ensemble['sigma'], results_ensemble['sigma_ensemble'] = means.mean(), torch.tensor([0.0]), means.std()
        if len(sigmas_forces)>0:  results_ensemble['dydx'], results_ensemble['sigma_forces'] = get_ensemble_stats(means_forces, sigmas_forces, self.mean_forces, self.std_forces)
        else: results_ensemble['dydx'], results_ensemble['sigma_forces'] = means_forces.mean(-2), means_forces.std(-2)
         
        print(means.mean())
        if len(sigmas)>0: print(sigmas.mean())
        print(results_ensemble['sigma_ensemble'])
 
        return results_ensemble
        
    def eval(self):
        self.model.eval()

def cdist_torch(vectors, vectors2):
    """ computes the coupled distances
    Args:
        vectors  is a N x D matrix
        vectors2 is a M x D matrix
        
        returns a N x M matrix
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors2)) + vectors.pow(2).sum(dim=1).view(-1, 1) + vectors2.pow(2).sum(dim=1).view(1, -1)
    return distance_matrix.abs()

        
class InducingPointLayer(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, n_in, n_points, inducing_points = None, trainable_points=True, trainable_gamma=True, use_bn=False):
        super().__init__()
        self.n_in = n_in
        self.use_bn = use_bn
        
        if use_bn: self.bn = nn.BatchNorm1d(n_in,eps=1e-3, affine=False, momentum=0.0)
        
        if inducing_points is None: inducing_points = torch.zeros(n_points, n_in).normal_().cuda()
        else: inducing_points = inducing_points.cuda()
        
        if trainable_points: self.register_parameter("inducing_points", Parameter(inducing_points))
        else: self.register_buffer("inducing_points", inducing_points)
        
        gamma = (torch.ones(n_in).float().cuda() + 0.25*torch.ones(n_in).float().normal_().cuda())*(n_in**(1/2))
        
        if trainable_gamma: self.register_parameter("gamma", Parameter(gamma))
        else: self.register_buffer("gamma", gamma)
        
    def forward(self, inputs):
#        pdb.set_trace()
        x, atom_mask = inputs["representation"], inputs["_atom_mask"].byte()
        
        B, A, _ = x.shape
        
        x = x[atom_mask].view(-1, self.n_in)
        mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True)+1e-5
        x = (x-mean)/std
        
        if self.use_bn: x = self.bn(x)
        x = cdist_torch(x/(self.gamma[None, ...].pow(2)+1e-3), self.inducing_points/(self.gamma[None, ...].pow(2)+1e-3))
#        x = cdist_torch(x, self.inducing_points)
        x = torch.exp(-x)
  
        #x_new = Variable(torch.zeros(B, A, len(self.inducing_points)).cuda())
        
        K_ind = cdist_torch(self.inducing_points/(self.gamma[None, ...].pow(2)+1e-3), self.inducing_points/(self.gamma[None, ...].pow(2)+1e-3))
        K_ind = torch.exp(-K_ind)
        K_inv = torch.inverse((K_ind+0.05*torch.eye(len(K_ind)).cuda()))
        
        x_new = Variable(torch.zeros(B, A, self.n_in).cuda())
        x_new[atom_mask] = torch.einsum('bn,nm,mi->bi', x, K_inv, self.inducing_points)*std + mean#x.mm(self.inducing_points)##x.mm(K_inv)#.mm(self.inducing_points)
        
#        pdb.set_trace()
        var = Variable(torch.zeros(B, A).cuda())
        var[atom_mask] = (torch.ones(x.shape[0]).float().cuda() - torch.einsum('bn,nm,bm->b', x, K_inv, x))# + self.sigma.pow(2).sum()#*std.pow(2)*self.sigma[None, ...].pow(2) #only for stationary rbf kernel, see wikipedia for formula
        
#        pdb.set_trace()
        
#        print(self.sigma.mean())
#        print((K_ind-torch.eye(len(self.inducing_points)).cuda()).max(1)[0].mean())
        
        return x_new, var
        
class InducingPointGP(nn.Module):
    def __init__(self, n_in, n_points, inducing_points = None, trainable_points=True, trainable_gamma=True, use_bn=False):
        super(InducingPointGP, self).__init__()
#        self.kernel = torch.eye(10)
#        self.mean = torch.zeros(n_in)
#        self.targets = torch.ones(10)
        gamma   = torch.ones(1, n_in)*(-5)
        sigma   = torch.ones(1)*(-3)
        targets = torch.ones(n_points).normal_()
        L     = torch.ones(n_points,n_points)
        alpha = torch.ones(n_points) 
        self.n_in  = n_in
        
        means   = torch.ones(n_in,1)
        stddevs = torch.ones(n_in,1)
        
        if inducing_points is None: inducing_points = torch.zeros(n_points, n_in).normal_().cuda()
        else: inducing_points = inducing_points.cuda()

        if targets is None: targets = torch.zeros(n_points, n_in).normal_().cuda()
        else: targets = targets.cuda()
        
        if trainable_points: self.register_parameter("inducing_points", Parameter(inducing_points))
        else: self.register_buffer("inducing_points", inducing_points)
                
        self.register_parameter("targets", Parameter(targets.cuda()))
        
        if trainable_gamma: self.register_parameter("gamma", Parameter(gamma.cuda()))
        else: self.register_buffer("gamma", gamma.cuda())
        
        self.register_parameter("sigma", Parameter(sigma.cuda()))
        
        self.register_buffer("L", L.cuda())
        self.register_buffer("alpha", alpha.cuda())
        self.register_buffer("means", means.cuda())
        self.register_buffer("stddevs", stddevs.cuda())        
                
    def forward(self,inputs):
        x, atom_mask = inputs["representation"], inputs["_atom_mask"].byte()
        
        B, A, _ = x.shape
        
        x = x[atom_mask].view(-1, self.n_in)
        mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True)+1e-5
        x = (x-mean)/std

        if self.training:
            K          = self.RBF_kernel(self.inducing_points,self.inducing_points) + torch.eye(len(self.inducing_points)).cuda()*(self.sigma.exp()+1e-3)
            self.L     = torch.cholesky(K, upper=False)
            self.alpha = torch.einsum("in,ij,j->n", torch.inverse(self.L), torch.inverse(self.L), self.targets)

#        pdb.set_trace()            

        K_xnew = self.RBF_kernel(self.inducing_points, x)

        v = torch.inverse(self.L).mm(K_xnew)
#        mean   = torch.einsum("in, i->n", K_xnew, self.alpha)
        var = torch.diagonal(self.RBF_kernel(x, x)) - torch.einsum("in,in->n",v,v)
        
        x_new = Variable(torch.zeros(B, A, len(self.inducing_points)).cuda())
        x_new[atom_mask] = K_xnew.transpose(0,1)
        #x_new[atom_mask] = mean

        var_new = Variable(torch.zeros(B, A).cuda())
        var_new[atom_mask] = var

        return x_new, var_new
            
    def RBF_kernel(self, x1, x2):
        return torch.exp(-cdist_torch(x1*self.gamma.exp(), x2*self.gamma.exp()))
        
    def Material_kernel(self, x1, x2):
        return material_distances(x1*self.gamma.exp(), x2*self.gamma.exp())
        
#    def likelihood(self):
#        K      = self.RBF_kernel(self.inducing_points,self.inducing_points) + torch.eye(len(self.data_points))*self.sigma
#        K_inv  = torch.inverse(K)        
#        return -1/2*(torch.einsum("ni,ij,mj->nm", self.targets, K_inv, self.targets) + logdet_torch(K) + self.n_in*math.log(2*math.pi))
        
        
class InducingPointBlock(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, n_in, n_points, out_net, inducing_points = None, trainable_points=True, trainable_gamma=True, use_bn=False):
        super().__init__()
        self.inducing_point_layer = InducingPointGP(n_in, n_points, inducing_points = inducing_points, trainable_points=trainable_points, trainable_gamma=trainable_gamma, use_bn=use_bn)
        self.out_net = nn.Linear(n_points,1)#out_net
        
    def forward(self, inputs):
        x, var = self.inducing_point_layer(inputs)
        return self.out_net(x), var, self.inducing_point_layer.gamma#self.out_net(x), var, self.inducing_point_layer.gamma
