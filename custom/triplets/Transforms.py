
import random
#from mendeleev import element
#from periodictable import elements
import numpy as np
from copy import deepcopy
import torch
import os

import pdb

elements_symbols = {'Na': 1, 'K' : 1, 'Rb': 1, 'Cs': 1,
                    'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
                    'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
                    'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 4,
                    'P' : 5, 'As': 5, 'Sb': 5, 'Bi': 5,
                    'O' : 6, 'S' : 6, 'Se': 6, 'Te': 6,
                    'Cl': 7, 'Br': 7, 'I' : 7}

#atoms_vol   = {i:element(int(i)).atomic_volume for i in np.arange(1,90)}
#name_to_int = {element(int(i)).symbol:i for i in np.arange(1,90)}

atoms_vol   = np.load(os.path.join(os.path.dirname(__file__),"mendeleev_atomic_radii_vol.npy")).item()
name_to_int = np.load(os.path.join(os.path.dirname(__file__), "mendeleev_name_to_int.npy")).item()

elements = {name_to_int[k]:v for k,v in elements_symbols.items()}

def get_similar_atoms(atom):
    if elements[atom] in [1, 2]:
        return [key for key in elements.keys() if elements[key] in [1,2]]
    if elements[atom] in [3, 4, 5]:
        return [key for key in elements.keys() if elements[key] in [3, 4, 5]]
    if elements[atom] in [6]:
        return [key for key in elements.keys() if elements[key] in [6]]
    return [atom]

def mutate_atom(atom, atom_numbers=None):
    #scale_factor_prototype = get_scale_factor(atom)
    vol_atoms = sum([atoms_vol[int(i)] for i in atom["_atomic_numbers"]])
    
    if atom_numbers is None:
        atom_numbers = atom["_atomic_numbers"].numpy()
        at_numbers_set = set(atom_numbers)
        for at in at_numbers_set:
            similar_atoms = get_similar_atoms(at)
            atom_numbers[atom_numbers==at] = random.choice(similar_atoms)
        
    atom["_atomic_numbers"] = torch.tensor(atom_numbers).type(atom["_atomic_numbers"].dtype)
    
    scale_factor_cell = (sum([atoms_vol[i] for i in atom_numbers])/vol_atoms)**(1/3)
    
    scale_factor_cell *= random.uniform(0.95,1.05)
    
    atom["_cell"] = atom["_cell"]*scale_factor_cell
    
    atom["_positions"] = atom["_positions"]*scale_factor_cell
    
    return atom, atom_numbers, scale_factor_cell

class MutateAtom(object):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, at_numbers=None, scalefactors = None):
        super(MutateAtom, self).__init__()
        if at_numbers is not None:
            self.at_numbers = at_numbers
            self.scalefactors = scalefactors
            self.first = False
        else:
            self.at_numbers = []
            self.scalefactors = []
            self.first = True
            
    def __call__(self, input, idx):
        if self.first:
            atom, at_numbers, scalefactor = mutate_atom(input)
            self.at_numbers.append(at_numbers)
            self.scalefactors.append(scalefactor)
        else:
            atom, _, _ = mutate_atom(input, self.at_numbers[idx], self.scalefactors[idx])
        
        return atom
        
    def reset(self):
        self.at_numbers = []
        self.scalefactors = []
        self.first = True 
        
class MutateAtomOnline(object):
    """Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, at_numbers=None, scalefactors = None):
        super(MutateAtomOnline, self).__init__()
        if at_numbers is not None:
            self.at_numbers = at_numbers
            self.scalefactors = scalefactors
            self.first = False
        else:
            self.at_numbers = []
            self.scalefactors = []
            self.first = True
            
    def __call__(self, input):
        atom, _, _ = mutate_atom(input)
        return atom
        
    def reset(self):
        self.at_numbers = []
        self.scalefactors = []
        self.first = True       