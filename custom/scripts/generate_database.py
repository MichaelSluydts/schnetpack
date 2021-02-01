import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.io import read,write
from ase.calculators.calculator import Calculator, FileIOCalculator
from copy import deepcopy
#from help_functions import *
#import sys,subprocessb

import pdb

energies = []
poslist = []
forcelist = []
typelist = []
rveclist = []
config = 0
atoms = []

EformationPerAtom = 'formation_energy_per_atom'
EPerAtom = 'energy_per_atom'
BandGap = 'band_gap'
TotalMagnetization = 'total_magnetization'

properties = [
    EformationPerAtom, EPerAtom, BandGap, TotalMagnetization
]

asedb_path = "/dev/shm/data/mp.db"

asedb = connect(asedb_path)

calculator = Calculator()
calculator.implemented_properties = ["forces", "energy"]

#atoms_template = read("Gepurebulk64.xyz",index=':',format='extxyz')[0]

#pdb.set_trace()

N_atoms = len(asedb)

for ind in range(N_atoms):
    row = asedb.get(ind+1)
    at  = row.toatoms()
    
    config += 1

    if config % 1000 == 0:
        print(config)
        
#    pdb.set_trace()
    energy = row.data["formation_energy_per_atom"]*len(at)
#    forces = at.get_forces()

    numbers = np.int32(at.numbers)

    positions = np.float32(at.positions)
    
    forces = 0*positions
    forces = np.float32(forces)
    energy = np.float32(energy)

    rvec = np.float32(at.cell)
    
    energies.append(energy)
    forcelist.append(np.atleast_2d(forces))
    rveclist.append(rvec)
    poslist.append(positions)
    typelist.append(numbers)
    
#    pdb.set_trace()
    
    at.set_calculator(deepcopy(calculator))
    at.get_calculator().atoms   = at
    at.get_calculator().results = {'energy': energy, 'forces': forces}
    
    atoms.append(at)
    
#     atoms_template.calculator.set_cell(rvec)
#     atoms_template.calculator.set_forces(forces)
#     atoms_template.calculator.set_positions(positions)
#     atoms_template.calculator.set_energy(energy)
#     atoms_template.calculator.set_atomic_numbers(numbers)
   

# for at in read(sys.argv[1],index=':',format='extxyz'):
#     config += 1

#     if config % 1000 == 0:
#         print(config)
    
#     energy = at.get_total_energy()
#     forces = at.get_forces()

#     numbers = np.int32(at.get_atomic_numbers())

#     positions = np.float32(at.get_positions())
#     forces = np.float32(forces)
#     energy = np.float32(energy)

#     rvec = np.float32(at.get_cell())
#     energy = energy 
#     energies.append(energy)
#     forcelist.append(forces)
#     rveclist.append(rvec)
#     poslist.append(positions)
#     typelist.append(numbers)
#     atoms.append(at)


#energies -= np.mean(energies)
#energies /= 64
#pdb.set_trace()

df = pd.DataFrame({'E' : energies, 'F' : forcelist, 'pos' : poslist, 'type' : typelist, 'rvec' : rveclist, 'atoms' : atoms})
df['Ebin'] = pd.qcut(df['E'].values,q=20,labels=range(20))
#print(df['Ebin'].head(10))
#print(df['Ebin'].value_counts())
train,test = train_test_split(df, test_size=0.20, random_state=42,stratify=df['Ebin'].values, shuffle = True)
train = train.sample(frac=1).reset_index(drop=True)
test = test.sample(frac=1).reset_index(drop=True)


#print(train['Ebin'].value_counts())
#print(test['Ebin'].value_counts())
write('train.xyz',train['atoms'].values,format='extxyz')
write('test.xyz',test['atoms'].values,format='extxyz')

train = train.iloc[0:3200]
test = test.iloc[0:3200]

write('train_short.xyz',train['atoms'].values,format='extxyz')
write('test_short.xyz',test['atoms'].values,format='extxyz')

exit()
