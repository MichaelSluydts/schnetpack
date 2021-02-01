# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:46:16 2018

@author: Michiel
"""

#from fastai.conv_learner import *
#from fastai.dataset import *
import schnetpack as spk
from schnetpack2.environment import SimpleEnvironmentProvider
import torch
import sys
import argparse
import logging
from shutil import copyfile, rmtree
from schnetpack2.datasets import MaterialsProject
from schnetpack2.utils import to_json, read_from_json, compute_params
import os
import mendeleev

from torch import nn, optim
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler

import numpy as np

import tqdm

import pdb

#torch.cuda.set_device(3)
torch.backends.cudnn.benchmark=True

db_PATH = r"/scratch/leuven/412/vsc41276"
model_PATH = r"/user/data/gent/gvo000/gvo00003/vsc41276/OQMD/MP_GAN"

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = list(m.children())
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))
    
def clear_tqdm():
    inst = getattr(tqdm.tqdm, '_instances', None)
    if not inst: return
    try:
        for i in range(len(inst)): inst.pop().close()
    except Exception:
        pass
    
def trange(*args, **kwargs):
    clear_tqdm()
    return tqdm.trange(*args, file=sys.stdout, **kwargs)

def is_half_tensor(v):
    return isinstance(v, torch.cuda.HalfTensor)

def to_np(v):
    '''returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.'''
    if isinstance(v, float): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if torch.cuda.is_available():
        if is_half_tensor(v): v=v.float()
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()

def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()

    ## command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument('--cuda', help='Set flag to use GPU(s)', action='store_true')
    cmd_parser.add_argument('--parallel',
                            help='Run data-parallel on all available GPUs (specify with environment variable'
                                 + ' CUDA_VISIBLE_DEVICES)', action='store_true')
    cmd_parser.add_argument('--batch_size', type=int,
                            help='Mini-batch size for training and prediction (default: %(default)s)',
                            default=32)
    ## training
    train_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    train_parser.add_argument('datapath', help='Path / destination of Materials Project dataset directory')
    train_parser.add_argument('modelpath', help='Destination for models and logs')
    train_parser.add_argument('--property', type=str,
                              help='Materials Project property to be predicted (default: %(default)s)',
                              default="formation_energy_per_atom", choices=MaterialsProject.properties)
    train_parser.add_argument('--apikey', help='API key for Materials Project (see https://materialsproject.org/open)')
    train_parser.add_argument('--seed', type=int, default=None, help='Set random seed for torch and numpy.')
    train_parser.add_argument('--overwrite', help='Remove previous model directory.', action='store_true')

    train_parser.add_argument('--split_path', help='Path / destination of npz with data splits',
                              default=None)
    train_parser.add_argument('--split', help='Split into [train] [validation] and use remaining for testing',
                              type=int, nargs=2, default=[None, None])
    train_parser.add_argument('--max_epochs', type=int, help='Maximum number of training epochs (default: %(default)s)',
                              default=5000)
    train_parser.add_argument('--lr', type=float, help='Initial learning rate (default: %(default)s)',
                              default=1e-4)
    train_parser.add_argument('--lr_patience', type=int,
                              help='Epochs without improvement before reducing the learning rate (default: %(default)s)',
                              default=50)
    train_parser.add_argument('--lr_decay', type=float, help='Learning rate decay (default: %(default)s)',
                              default=0.5)
    train_parser.add_argument('--lr_min', type=float, help='Minimal learning rate (default: %(default)s)',
                              default=1e-6)

    train_parser.add_argument('--logger', help='Choose logger for training process (default: %(default)s)',
                              choices=['csv', 'tensorboard'], default='csv')
    train_parser.add_argument('--log_every_n_epochs', type=int,
                              help='Log metrics every given number of epochs (default: %(default)s)',
                              default=1)

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path of MaterialsProject dataset directory')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--apikey', help='API key for Materials Project (see https://materialsproject.org/open)',
                             default=None)
    eval_parser.add_argument('--split', help='Evaluate trained model on given split',
                             choices=['train', 'validation', 'test'], default=['test'], nargs='+')

    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument('--aggregation_mode', type=str, default='avg', choices=['sum', 'avg'],
                              help=' (default: %(default)s)')

    #######  SchNet  #######
    schnet_parser = argparse.ArgumentParser(add_help=False, parents=[model_parser])
    schnet_parser.add_argument('--features', type=int, help='Size of atom-wise representation (default: %(default)s)',
                               default=64)
    schnet_parser.add_argument('--interactions', type=int, help='Number of interaction blocks (default: %(default)s)',
                               default=2)
    schnet_parser.add_argument('--cutoff', type=float, default=5.,
                               help='Cutoff radius of local environment (default: %(default)s)')
    schnet_parser.add_argument('--num_gaussians', type=int, default=25,
                               help='Number of Gaussians to expand distances (default: %(default)s)')

    ## setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode', help='Command-specific arguments')
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser('train', help='Training help')
    subparser_eval = cmd_subparsers.add_parser('eval', help='Training help')

    train_subparsers = subparser_train.add_subparsers(dest='model', help='Model-specific arguments')
    train_subparsers.required = True

    subparser_export = cmd_subparsers.add_parser('export', help='Export help')
    subparser_export.add_argument('modelpath', help='Path of stored model')
    subparser_export.add_argument('destpath', help='Destination path for exported model')

    train_subparsers.add_parser('schnet', help='SchNet help', parents=[train_parser, schnet_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model', help='Model-specific arguments')
    eval_subparsers.add_parser('schnet', help='SchNet help', parents=[eval_parser, schnet_parser])

    return main_parser

def get_environment(n_atoms, grid=None):

    if n_atoms == 1:
        neighborhood_idx = -torch.ones((1, 1), dtype=torch.float32)
        offsets = torch.zeros((n_atoms, 1, 3), dtype=torch.float32)
    else:
        neighborhood_idx = torch.arange(n_atoms, dtype=torch.float32).unsqueeze(0).repeat(n_atoms, 1)
        neighborhood_idx = neighborhood_idx[~torch.eye(n_atoms, dtype=torch.long).byte()].view(n_atoms, n_atoms - 1).long()

        if grid is not None:
            n_grid = grid.shape[0]
            neighborhood_idx = torch.concat([neighborhood_idx, -torch.ones((n_atoms, 1))], 1)
            grid_nbh = torch.tile(torch.arange(n_atoms, dtype=torch.float32).unsqueeze(-1), (n_grid, 1))
            neighborhood_idx = torch.concat([neighborhood_idx, grid_nbh], 0)

        offsets = torch.zeros((neighborhood_idx.shape[0], neighborhood_idx.shape[1], 3), dtype=torch.float32)
    return neighborhood_idx, offsets

class Structure:
    """
    Keys to access structure properties loaded using `schnetpack2.data.AtomsData`
    """
    Z = '_atomic_numbers'
    atom_mask = '_atom_mask'
    R = '_positions'
    cell = '_cell'
    neighbors = '_neighbors'
    neighbor_mask = '_neighbor_mask'
    cell_offset = '_cell_offset'
    neighbor_pairs_j = '_neighbor_pairs_j'
    neighbor_pairs_k = '_neighbor_pairs_k'
    neighbor_pairs_mask = '_neighbor_pairs_mask'
    
class Convert_to_graph(nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, atoms, positions, cell):    
            # extract properties
            properties = {}
    
            # extract/calculate structure
            properties[Structure.Z] = atoms
    
            properties[Structure.R] = positions
    
            properties[Structure.cell] = cell
            
            properties[Structure.cell_offset]   = torch.zeros(len(atoms), *[int(ss) for ss in (atoms.shape[1], atoms.shape[1]-1, 3)])
            properties[Structure.neighbor_mask] = torch.zeros(len(atoms), *[int(ss) for ss in (atoms.shape[1], atoms.shape[1]-1)], dtype=torch.float)
            properties[Structure.neighbors]     = torch.zeros(len(atoms), *[int(ss) for ss in (atoms.shape[1], atoms.shape[1]-1)], dtype=torch.long)
            properties[Structure.atom_mask]     = torch.zeros(len(atoms), atoms.shape[1], dtype=torch.float)
#            properties['_idx'] = torch.LongTensor(np.array([idx], dtype=np.int))

            for k, _ in enumerate(positions): 
                nbh, offsets = get_environment(len(atoms[k]))
#                nbh = torch.LongTensor(nbh_idx.astype(np.int))
#                offsets = torch.FloatTensor(offsets.astype(np.float32))
                
                properties[Structure.cell_offset][k] = offsets
                
                # add mask
                shape = nbh.size()
                s = (k,) + tuple([slice(0, d) for d in shape])
                mask = (nbh >= 0).long()
                properties[Structure.neighbor_mask][s] = mask
                properties[Structure.neighbors][s] = nbh * mask
            
                shape = atoms[k].size()
                s = (k,) + tuple([slice(0, d) for d in shape])
                properties[Structure.atom_mask][s] = (atoms[k] > 0)
                   
    
            return properties
        
def create_noisy_poscar(bs):
#    pdb.set_trace()
    n_elements = [np.random.randint(2,6) for b in range(bs)]
    n_atoms    = np.random.randint(2,100)
    elements   = [np.random.choice(np.arange(100),n_elements[b]) for b in range(bs)]
    atoms      = np.array([sorted([elements[b][np.random.randint(0,n_elements[b])] for atom in range(n_atoms)]) for b in range(bs)])
    
#    atoms      = torch.zeros(bs, n_atoms, dtype = torch.long)
    positions  = torch.zeros(bs, n_atoms, 3, dtype = torch.float)
    cell       = torch.zeros(bs, 3, 3, dtype = torch.float)
    
    #atoms.random_(1, 100)
    atoms      = torch.from_numpy(atoms).long()
    positions.uniform_(0,100)
    cell.uniform_(0,100)
    
    cell += torch.eye(3).repeat((bs,1,1))*cell.sum(-1).unsqueeze(-1)
    
    return atoms.cuda(), positions.cuda(), cell.cuda()
    
def get_model(args, atomref=None, mean=None, stddev=None, train_loader=None, parallelize=False):
    if args.model == 'schnet':
        representation = spk.representation.SchNet(args.features, args.features, args.interactions,
                                                   args.cutoff, args.num_gaussians, normalize_filter=True)
        atomwise_output = spk.atomistic.Energy(args.features,
                                                mean=mean, stddev=stddev, create_graph = False, return_force = False,
                                                atomref=atomref, train_embeddings=True)
        model = spk.atomistic.AtomisticModel(representation, atomwise_output)
    else:
        raise ValueError('Unsupported model class:', args.model)

    if parallelize:
        model = nn.DataParallel(model)

    logging.info(f"The model you built has: {compute_params(model)} parameters")

    return model
    
        
class ConvBlock(nn.Module):
    def __init__(self, ni, no, ks, stride, bn=True, pad=None, activation = nn.LeakyReLU(negative_slope=-1.0, inplace=True)):
        super().__init__()
        if pad is None: pad = ks//2//stride
        self.conv = nn.Conv1d(ni, no, ks, stride, padding=pad, bias=True)
        self.bn = nn.BatchNorm1d(no) if bn else None
        self.relu = activation
    
    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x) if self.bn else x
    
class DCGAN_G(nn.Module):
    def __init__(self, ks = 25, n_extra_layers=0):
        super().__init__()
        
        self.ks = ks

        self.initial_pos = ConvBlock(3, 3, self.ks, 1, bn=False, pad=True, activation = nn.Sigmoid())
        self.extra_pos = nn.ModuleList([ConvBlock(3, 3, self.ks, 1, pad=True, activation = nn.Sigmoid())
                                    for t in range(n_extra_layers)])
        self.final_pos = ConvBlock(3, 3, self.ks, 1, pad=True, bn=False, activation = nn.Sigmoid())
                                    
        self.initial_cell = ConvBlock(3, 3, self.ks, 1, pad=True, bn=False)
        self.extra_cell   = nn.ModuleList([ConvBlock(3, 3, self.ks, 1, pad=True)
                                    for t in range(n_extra_layers)])
        self.final_cell = ConvBlock(3, 3, self.ks, 1, pad=True, bn=False)
    
    def forward(self, atoms, positions, cells):
        bs = len(atoms)
    
        positions = self.initial_pos(positions.transpose(1,2))
        cells     = self.initial_cell(cells.transpose(1,2))
        
        cells = cells + torch.eye(3).repeat((bs,1,1)).cuda()*cells.sum(-1).unsqueeze(-1)
        
        positions = positions.transpose(1,2).bmm(cells.transpose(1,2)).transpose(1,2)
        
        for l_pos, l_cell in zip(self.extra_pos, self.extra_cell):
            positions = l_pos(positions)
            cells     = l_cell(cells)
            cells = cells + torch.eye(3).repeat((bs,1,1)).cuda()*cells.sum(-1).unsqueeze(-1)
            
            positions = positions.transpose(1,2).bmm(cells.transpose(1,2)).transpose(1,2)
        
        positions = self.final_pos(positions)
        
        cells     = self.final_cell(cells)
        
        cells = cells + torch.eye(3).repeat((bs,1,1)).cuda()*cells.sum(-1).unsqueeze(-1)
        
        positions = positions.transpose(1,2).bmm(cells.transpose(1,2))
        
        return atoms, 100*positions, 100*cells.transpose(1,2)

class DCGAN_D(nn.Module):
    def __init__(self, schnet, property):
        super().__init__()
        self.schnet = schnet
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.property = property

    def forward(self, input):
        x = self.schnet(input)
        return self.relu(x['y']).mean(0).view(1)
    
def train(niter, train_loader, first=True):
    gen_iterations = 0
    for epoch in trange(niter):
        netD.train(); netG.train()
        data_iter = iter(train_loader)
        i,n = 0,len(train_loader)
        with tqdm.tqdm(total=n) as pbar:
            while i < n:
                set_trainable(netD, True)
                set_trainable(netG, False)
                d_iters = 100 if (first and (gen_iterations < 25) or (gen_iterations % 500 == 0)) else 5
                j = 0
                while (j < d_iters) and (i < n):
#                    pdb.set_trace()
                    j += 1; i += 1
                    for p in netD.parameters(): p.data.clamp_(-0.01, 0.01)
                    real = next(data_iter)
#                    print(real[Structure.R].shape)
#                    print(real[Structure.cell].shape)
#                    print(real[Structure.Z].shape)
                    real_loss = netD({k:v.cuda() for k,v in real.items()})
                    fake = netG(*create_noisy_poscar(real[list(real.keys())[0]].shape[0]))
                    fake = converter(*fake)
#                    print(fake[Structure.R].shape)
#                    print(fake[Structure.cell].shape)
#                    print(fake[Structure.Z].shape)
                    fake_loss = netD({k:v.cuda() for k,v in fake.items()})
                    netD.zero_grad()
                    lossD = (real_loss-fake_loss)
                    lossD.backward()
                    optimizerD.step()
                    pbar.update()

                set_trainable(netD, False)
                set_trainable(netG, True)
                netG.zero_grad()
#                pdb.set_trace()
                fake = converter(*netG(*create_noisy_poscar(real[list(real.keys())[0]].shape[0])))
                lossG = netD({k:v.cuda() for k,v in fake.items()})
                lossG.backward()
                optimizerG.step()
                gen_iterations += 1

        torch.save(netG.state_dict(), model_PATH+'/netG_2.h5')
        torch.save(netD.state_dict(), model_PATH+'/netD_2.h5')
            
        print(f'Loss_D {to_np(lossD)}; Loss_G {to_np(lossG)}; '
              f'D_real {to_np(real_loss)}; Loss_D_fake {to_np(fake_loss)}')

def write_poscar(atoms, positions, cells, save_dir, epochs = 1):

    PER_DICT = {i+1:mendeleev.element(i+1).symbol for i in range(100)}

    for ind, (atom, position, cell) in enumerate(zip(atoms, positions, cells)):
        poscar  = "generated after " + str(epochs) + " epochs\n"
        poscar += "1\n"
        
        for line in cell:
            poscar += "\t".join([str(at) for at in line])+"\n"
        elements, elements_num = np.unique(atom, return_counts = True)
        poscar += "\t".join([PER_DICT[element+1] for element in elements])+"\n"
        poscar += "\t".join([str(element_num) for element_num in elements_num])+"\n"
        poscar += "Cartesian\n"
        for line in position:
            poscar += "\t".join([str(pos) for pos in line])+"\n"
        with open(save_dir+"/GAN_"+str(epochs)+"_"+str(ind)+".vasp", "w") as file:
            file.write(poscar)
            
bs,sz,nz = 32,200,10

parser = get_parser()
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")
argparse_dict = vars(args)
jsonpath = os.path.join(args.modelpath, 'args.json')

if args.mode == 'train':
    if args.overwrite and os.path.exists(args.modelpath):
        logging.info('existing model will be overwritten...')
        rmtree(args.modelpath)

    if not os.path.exists(args.modelpath):
        os.makedirs(args.modelpath)

    to_json(jsonpath, argparse_dict)

    spk.utils.set_random_seed(args.seed)
    train_args = args
else:
    train_args = read_from_json(jsonpath)

# will download MaterialsProject if necessary
mp = spk.datasets.MaterialsProject(args.datapath, args.cutoff, apikey=args.apikey, download=True,
                                   properties=[train_args.property])

# splits the dataset in test, val, train sets
split_path = os.path.join(args.modelpath, 'split.npz')
if args.mode == 'train':
    if args.split_path is not None:
        copyfile(args.split_path, split_path)

data_train, data_val, data_test = mp.create_splits(*train_args.split, split_file=split_path)

train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size, sampler=RandomSampler(data_train),
                                    num_workers=4, pin_memory=True)
val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size, num_workers=2, pin_memory=True)

if args.mode == 'train':
    mean, stddev = train_loader.get_statistics(train_args.property, False)
    logging.info('Training set statistics: mean=%.3f, stddev=%.3f' % (mean.numpy(), stddev.numpy()))
else:
    mean, stddev = None, None

model = get_model(train_args, atomref=None, mean=mean, stddev=stddev, train_loader=train_loader,
                  parallelize=args.parallel).to(device)

#mp = spk.datasets.MaterialsProject(args.datapath, args.cutoff, apikey=args.apikey, download=True,
#                                   properties=[train_args.property], environment_provider = SimpleEnvironmentProvider())

train_loader = spk.data.AtomsLoader(mp, batch_size=bs, sampler=RandomSampler(mp),
                                    num_workers=4, pin_memory=True)

netG = DCGAN_G(5).cuda()

converter = Convert_to_graph()

netD = DCGAN_D(model, train_args.property).cuda()

optimizerD = optim.RMSprop(netD.parameters(), lr = 1e-4)
optimizerG = optim.RMSprop(netG.parameters(), lr = 1e-4)

fixed_noise = create_noisy_poscar(10)

netD.eval(); netG.eval();
fake = netG(*fixed_noise)
fake = [inp.detach().cpu().numpy() for inp in fake]

write_poscar(*fake, "GAN_creations", epochs = 0)

netD.train(); netG.train();
train(5, train_loader, False)

netD.eval(); netG.eval();
fake = netG(*fixed_noise)
fake = [inp.detach().cpu().numpy() for inp in fake]

write_poscar(*fake, "GAN_creations", epochs = 1)

set_trainable(netD, True)
set_trainable(netG, True)
optimizerD = optim.RMSprop(netD.parameters(), lr = 1e-5)
optimizerG = optim.RMSprop(netG.parameters(), lr = 1e-5)

netD.train(); netG.train();
train(30, train_loader, False)

torch.save(netG.state_dict(), model_PATH+'/netG_2.h5')
torch.save(netD.state_dict(), model_PATH+'/netD_2.h5')

netD.eval(); netG.eval();
fake = netG(*fixed_noise)
fake = [inp.detach().cpu().numpy() for inp in fake]

write_poscar(*fake, "GAN_creations", epochs = 2)
