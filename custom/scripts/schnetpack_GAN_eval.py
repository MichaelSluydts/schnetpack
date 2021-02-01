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
import mendeleev

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
        
def generate_and_save_crystals(netD, netG, n_materials, rnd_dl, savedir, thresshold):
    data_iter_rnd = iter(rnd_dl)
    
    atoms = []
    positions = []
    cells = []
    
    with torch.no_grad():
        while len(atoms) < n_materials:
            fake_input = next(data_iter_rnd)
            fake_input = {k:Variable(v.cuda()) for k, v in fake_input.items()}
            fake = netG(fake_input)
            fake_loss = netD(fake)["y"].detach().cpu().numpy()
            
            indices_pass = np.where(fake_loss<thresshold)[0]
            
            print(fake_loss)
            
            for ind in indices_pass:
                mask = fake["_atom_mask"][ind].detach().cpu().numpy()
                n_atoms = int(sum(mask))
                atoms.append(fake["_atomic_numbers"].detach().cpu().numpy()[ind,:n_atoms])
                positions.append(fake["_positions"].detach().cpu().numpy()[ind,:n_atoms])
                cells.append(fake["_cell"].detach().cpu().numpy()[ind,:n_atoms])
                
            print(len(atoms))
                
    write_poscar(atoms, positions, cells, savedir)
            

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
args.savedir = os.path.join(os.environ[r"VSC_SCRATCH"],"GAN_creations")

if not os.path.isdir(args.savedir):
    os.makedirs(args.savedir)

device = torch.device("cuda")
train_args = args
#spk.utils.set_random_seed(args.seed)

env = ASEEnvironmentProvider(args.cutoff)
#name = r'/dev/shm/data/train' + sys.argv[1]
name = r'/dev/shm/data/mp'
data_train = schnetpack2.datasets.matproj.MaterialsProject(name, args.cutoff, properties = ["formation_energy_per_atom"], subset=np.arange(3000))
#name = r'/dev/shm/data/bulkVtrain3200'
#data_train = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])
#name = r'/dev/shm/data/test' + sys.argv[1]
#name = r'/dev/shm/data/bulkVtest3200'
#data_val = schnetpack2.custom.datasets.extxyz1.ExtXYZ(name + '.db',name + '.xyz',environment_provider = env, properties=['energy','forces'])

data_random = schnetpack2.custom.datasets.random_structure.RandomCrystal(num_elements=4, len_dataset=1000*len(data_train), max_elements=90, max_num_atoms=10,
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

netD.load_state_dict(torch.load(os.path.join(args.modelpath, "netD_model")))
netG.load_state_dict(torch.load(os.path.join(args.modelpath, "netG_model")))

netD.eval()
netG.eval()

n_materials = 100
threshold =  0#mean.item()

generate_and_save_crystals(netD, netG, n_materials, random_loader, args.savedir, threshold)