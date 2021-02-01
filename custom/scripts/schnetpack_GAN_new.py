import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable

import schnetpack as spk
from schnetpack2.environment import ASEEnvironmentProvider
from schnetpack2.datasets import MaterialsProject
from schnetpack2.utils import to_json, read_from_json, compute_params

import schnetpack2.custom.data
import schnetpack2.custom.datasets.extxyz1
import schnetpack2.custom.datasets.random_structure
import schnetpack2.custom.representation

from tqdm import tqdm, trange

import numpy as np
import os
import sys
import logging
import pdb

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def get_model(args, atomref=None, mean=None, stddev=None, parallelize=False):
    if args.model == 'schnet':
        representation = spk.custom.representation.SchNet(
            n_atom_basis=args.features, 
            n_filters=args.features, 
            n_interactions=args.interactions, 
            cutoff=args.cutoff, 
            n_gaussians=args.num_gaussians,
            normalize_filter=False, 
            coupled_interactions=False,
            return_intermediate=False, 
            max_z=args.maxz, 
            cutoff_network=schnetpack2.nn.cutoff.CosineCutoff, 
            filter_network_type="original",
            n_expansion_coeffs=args.num_gaussians,
            trainable_gaussians=False,
            distance_expansion=None,
            sc=args.sc,
            start = args.start,
            bn=False,p=0,debug=False, attention=0)
        atomwise_output = spk.custom.atomistic.Energy(args.features, mean=mean, aggregation_mode='avg', stddev=stddev,
                                                atomref=atomref, return_force=False, create_graph=False, train_embeddings=True, n_layers=args.outlayers,bn=False,p=0.0)
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)
 
    model = nn.DataParallel(model)

    logging.info(f"The model you built has: {compute_params(model)} parameters")
    
    return model

class DCGAN_D(nn.Module):
    def __init__(self, args, atomref=None, mean=None, stddev=None, parallelize=False):
        super().__init__()
        self.model = get_model(args, atomref=atomref, mean=mean, stddev=stddev, parallelize=parallelize)
        
    def forward(self, input):
        x = self.model(input)
        return x
        
class DCGAN_G(nn.Module):
    def __init__(self, args, atomref=None, parallelize=False):
        super().__init__()
        self.representation = spk.custom.representation.SchNet(
                                                    n_atom_basis=args.features, 
                                                    n_filters=args.features, 
                                                    n_interactions=args.interactions, 
                                                    cutoff=args.cutoff, 
                                                    n_gaussians=args.num_gaussians,
                                                    normalize_filter=False, 
                                                    coupled_interactions=False,
                                                    return_intermediate=False, 
                                                    max_z=args.maxz, 
                                                    cutoff_network=schnetpack2.nn.cutoff.CosineCutoff, 
                                                    filter_network_type="original",
                                                    n_expansion_coeffs=args.num_gaussians,
                                                    trainable_gaussians=False,
                                                    distance_expansion=None,
                                                    sc=args.sc,
                                                    start = args.start,
                                                    bn=False,p=0,debug=False, attention=0)
                                                    
        self.predictor = spk.custom.atomistic.MultiOutput(3, 9, mean=None, aggregation_mode='avg', stddev=None, n_neurons = [3,6,9],
                               atomref=atomref, return_force=False, create_graph=False, train_embeddings=True, n_layers=args.outlayers,bn=False,p=0.0)
        
        self.final_dense = nn.Linear(args.features,3)
        self.activation  = nn.Sigmoid()#nn.Softsign()
                               
        self.requires_dr = True

    def forward(self, inputs):
        if self.requires_dr:
            inputs["_positions"].requires_grad_()
        
        t = self.representation(inputs)
        
        #inputs['representation'] = t * inputs["_atom_mask"][...,None]
        
        t = self.final_dense(t)
        t = self.activation(10*t)#(self.activation(t)+1)/2#torch.sigmoid(10*t)

#        if type(t) != tuple:
#            t = t, None
#
#        inputs['representation'], _ = t
        
#        cell = self.predictor(inputs)
        
        inputs["_positions"] = ((inputs["_positions"].bmm(torch.inverse(inputs["_cell"])) + t)%1)
        inputs['representation'] = inputs["_positions"].bmm(inputs["_cell"])
        
        cell = self.predictor(inputs)
        
        num_atoms = inputs["_atom_mask"].sum(-1)
        
        inputs["_cell"] = 3*(num_atoms[:,None,None])**(1/3)*(torch.eye(3)[None,...].cuda() + cell["y"].view(-1,3,3))#2*(inputs["_positions"].shape[1])**(1/3)*(torch.eye(3)[None,...].cuda() + cell["y"].exp().view(-1,3,3))
        inputs["_positions"] = inputs["_positions"].bmm(inputs["_cell"])

        return inputs
    
def train(niter, trn_dl, rnd_dl, first=True):
    gen_iterations = 0
    for epoch in trange(niter):
        netD.train(); netG.train()
        data_iter = iter(trn_dl)
        data_iter_rnd = iter(rnd_dl)
        i,n = 0,len(trn_dl)
        with tqdm(total=n) as pbar:
            while i < n:
                set_trainable(netD, True)
                set_trainable(netG, False)
                d_iters = 100 if (first and (gen_iterations < 25) or (gen_iterations % 500 == 0)) else 5
                j = 0
                while (j < d_iters) and (i < n):
                    j += 1; i += 1
                    #for p in netD.parameters(): p.data.clamp_(-0.01, 0.01)# check improvement: WGAN-GP: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
                    inputs = next(data_iter)
                    inputs = {k:Variable(v.cuda()) for k, v in inputs.items()}
                    real_loss = netD(inputs)
                    fake_input = next(data_iter_rnd)
                    print(fake_input["_neighbor_mask"].shape)
                    fake_input = {k:Variable(v.cuda()) for k, v in fake_input.items()}
                    fake = netG(fake_input)
                    print(fake["_neighbor_mask"].shape)
                    fake_loss = netD(fake)
                    netD.zero_grad()
                    lossD = 1e-1*(real_loss["y"]-fake_loss["y"][:len(real_loss["y"])]).mean() + (real_loss["y"]-inputs["formation_energy_per_atom"]).pow(2).mean()
                    lossD.backward()
                    optimizerD.step()
                    pbar.set_description("Loss D {:08.2f}".format(lossD.item()))
                    del inputs
                    del fake_input
                    pbar.update()

                set_trainable(netD, False)
                set_trainable(netG, True)
                netG.zero_grad()
                fake_input = next(data_iter_rnd)
                fake_input = {k:Variable(v.cuda()) for k, v in fake_input.items()}
                lossG = netD(netG(fake_input))["y"].mean(0).view(1)
                lossG.backward()
                optimizerG.step()
                gen_iterations += 1
                pbar.set_description("Loss G {:08.2f}".format(lossG.item()))
                del fake_input
                pbar.update()
            
        print(f'Loss_D {lossD.item()}; Loss_G {lossG.item()}; '
              f'D_real {real_loss["y"].mean().item()}; Loss_D_fake {fake_loss["y"].mean().item()}')

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size, real_data.nelement()/args.batch_size).contiguous().view(args.batch_size, 3, 32, 32)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

torch.backends.cudnn.benchmark=True

class Args:
    def __init__(self):
        self.cuda = True
    
print(torch.cuda.device_count())
args = Args()
args.cuda = True
args.parallel = False
args.batch_size = int(sys.argv[1])*torch.cuda.device_count()
args.property = 'forces'
args.datapath = ''
args.modelpath = os.environ[r"VSC_SCRATCH"]
args.seed = 1337
args.overwrite = False
args.split_path = None
args.split = [65000,7000]

args.features = 128
args.interactions = [2,2,3]
args.cutoff = 5.
args.num_gaussians  = 64
args.start = 0.0
args.model = 'schnet'
args.sc=True
args.maxz = 100
args.outlayers = 5

device = torch.device("cuda")
train_args = args
#spk.utils.set_random_seed(args.seed)

env = ASEEnvironmentProvider(args.cutoff)
#name = r'/dev/shm/data/train' + sys.argv[1]

name = r'/dev/shm/data/mp'
data_train = schnetpack2.datasets.matproj.MaterialsProject(name, args.cutoff, properties = ["formation_energy_per_atom"], subset=np.random.choice(80000, size=10000, replace=False))
#name = r'/dev/shm/data/bulkVtrain3200'
#data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
#name = r'/dev/shm/data/test' + sys.argv[1]
#name = r'/dev/shm/data/bulkVtest3200'
#data_val = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])

data_random = schnetpack2.custom.datasets.random_structure.RandomCrystal(num_elements=4, len_dataset=len(data_train), max_elements=90, max_num_atoms=10,
                                                     properties=[], environment_provider=env)

train_loader = schnetpack2.custom.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                    num_workers=9*torch.cuda.device_count(), pin_memory=True)
#val_loader = schnetpack2.custom.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)

random_loader = schnetpack2.custom.data.AtomsLoader(data_random, batch_size=args.batch_size, num_workers=9*torch.cuda.device_count(), pin_memory=True)

mean, stddev = train_loader.get_statistics('formation_energy_per_atom', False)

print(mean)
print(stddev)

#pdb.set_trace()

netD = DCGAN_D(args, atomref=None, mean=torch.FloatTensor([mean]), stddev=torch.FloatTensor([stddev]), parallelize=args.parallel)
netG = DCGAN_G(args, atomref=None, parallelize=args.parallel)

if args.cuda:
    netD.cuda()
    netG.cuda()

optimizerD = optim.RMSprop(netD.parameters(), lr = float(sys.argv[3]))
optimizerG = optim.RMSprop(netG.parameters(), lr = float(sys.argv[3]))

train(int(sys.argv[2]), train_loader, random_loader, False)

set_trainable(netD, True)
set_trainable(netG, True)
optimizerD = optim.RMSprop(netD.parameters(), lr = float(sys.argv[3])/10)
optimizerG = optim.RMSprop(netG.parameters(), lr = float(sys.argv[3])/10)

train(int(sys.argv[2]), train_loader, random_loader, False)

torch.save(netD.state_dict(), os.path.join(args.modelpath, "netD_model"))
torch.save(netG.state_dict(), os.path.join(args.modelpath, "netG_model"))