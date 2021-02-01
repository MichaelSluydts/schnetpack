import argparse
import logging
import os
import sys
from shutil import copyfile, rmtree

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.sampler import RandomSampler

import schnetpack as spk
from schnetpack2.data import AtomsData
from schnetpack2.utils import to_json, read_from_json, compute_params
from schnetpack2.custom.environment import MaartenEnvironmentProvider
from schnetpack2.environment import ASEEnvironmentProvider
import ctypes as C

lib = C.CDLL('./cell_list.so')

atom_list = lib.atom_list
atom_list.argtypes = [np.ctypeslib.ndpointer(dtype = np.float32, flags="C_CONTIGUOUS"), np.ctypeslib.ndpointer(dtype = np.float32, flags="C_CONTIGUOUS"), C.c_float, C.c_int, C.POINTER(C.c_int)]
atom_list.restype = C.POINTER(C.c_float)

def make_nd_array(c_pointer, shape, dtype=np.float32, order='C', own_data=True):
    arr_size = np.prod(shape[:]) * np.dtype(dtype).itemsize 
    if sys.version_info.major >= 3:
        buf_from_mem = C.pythonapi.PyMemoryView_FromMemory
        buf_from_mem.restype = C.py_object
        buf_from_mem.argtypes = (C.c_void_p, C.c_int, C.c_int)
        buffer = buf_from_mem(c_pointer, arr_size, 0x100)
    else:
        buf_from_mem = C.pythonapi.PyBuffer_FromMemory
        buf_from_mem.restype = C.py_object
        buffer = buf_from_mem(c_pointer, arr_size)
    arr = np.ndarray(tuple(shape[:]), dtype, buffer, order=order)
    if own_data and not arr.flags.owndata:
        return arr.copy()
    else:
        return arr

def py_list_cell(at_coords, at_cell, rcut):
    at_coords = np.ascontiguousarray(at_coords.T, dtype =np.float32)
    at_cell   = np.ascontiguousarray(at_cell, dtype =np.float32)
    N = C.c_int(len(at_coords))
    max_neighbours = C.c_int(-1)
    output_ptr = atom_list(C.byref(at_coords), C.byref(at_cell), C.c_float(rcut), N, C.byref(max_neighbours))
    arr_shape = (N.value,max_neighbours.value, 4)

    arr_temp = np.ctypeslib.as_array(output_ptr, shape = arr_shape)

    return arr_temp[:,:,0], arr_temp[:,:,1:]#[output_ptr[i] for i in range(N.value*max_neighbours.value*4)]#np.ctypeslib.as_array(output_ptr, shape = arr_shape) #make_nd_array(output_ptr, arr_shape)

batch_size = 16
cutoff     = 5.0
apikey     = None
datapath   = "/scratch/leuven/412/vsc41276/mp.db"
property   = "formation_energy_per_atom"

mp_ASE = AtomsData(datapath, properties=[property], environment_provider= ASEEnvironmentProvider(cutoff))

mp_MAARTEN = AtomsData(datapath, properties=[property], environment_provider = MaartenEnvironmentProvider(cutoff))

ASE_loader = spk.data.AtomsLoader(mp_ASE, batch_size=batch_size, sampler=RandomSampler(mp_ASE),
                                    num_workers=36, pin_memory=True)

MAARTEN_loader = spk.data.AtomsLoader(mp_MAARTEN, batch_size=batch_size, sampler=RandomSampler(mp_ASE),
                                    num_workers=36, pin_memory=True)
