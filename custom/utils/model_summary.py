import torch as th
from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import numpy as np
from schnetpack2.data import Structure

import pdb

def make_input(device, batch_size = 10, N_atoms = 5):
        inputs = {}
        
        inputs[Structure.Z]             = th.randint(1, 30, (batch_size, N_atoms)).long()
        inputs[Structure.R]             = th.rand(batch_size, N_atoms, 3)
        inputs[Structure.cell]          = th.eye(3).unsqueeze(0).expand(batch_size,3,3)
        inputs[Structure.cell_offset]   = th.randint(-1, 2, (batch_size, N_atoms, N_atoms-1,3)).float()
        inputs[Structure.neighbors]     = th.randint(0, N_atoms, (batch_size, N_atoms,N_atoms-1)).long()
        inputs[Structure.neighbor_mask] = th.zeros_like(inputs[Structure.neighbors]).float()
        inputs[Structure.atom_mask]     = th.zeros_like(inputs[Structure.Z]).float()

        return {k:Variable(v.to(device)) for k,v in inputs.items()}

def summary(model, device):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                
                if (isinstance(input, list) or isinstance(input, tuple)) and isinstance(input[0], th.Tensor):
                  summary[m_key]['input_shape'] = list(input[0].size())
                  summary[m_key]['input_shape'][0] = -1
                elif (isinstance(input, list) or isinstance(input, tuple)) and isinstance(input[0], dict):
                  summary[m_key]['input_shape'] = list(input[0][Structure.neighbors].size())
                  summary[m_key]['input_shape'][0] = -1
                elif isinstance(input, tuple) and isinstance(input[0], list) and isinstance(input[0][0], th.Tensor):#only for SC layer
                  summary[m_key]['input_shape'] = list(input[0][0].size())
                  summary[m_key]['input_shape'][0] = -1
                if not isinstance(output, dict):
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = -1
                else:
                  summary[m_key]['output_shape'] = list(output['y'].size())
                  summary[m_key]['output_shape'][0] = -1
                  
                params = 0
                if hasattr(module, 'weight'):
                    params += th.prod(th.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias') and module.bias is not None:
                    params +=  th.prod(th.LongTensor(list(module.bias.size())))
                if hasattr(module, 'offsets'):
                    params += th.prod(th.LongTensor(list(module.offsets.size())))
                    if module.offsets.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False                
                if hasattr(module, 'width'):
                    params += th.prod(th.LongTensor(list(module.width.size())))
                    if module.width.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False        
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
                
#        dtype = th.cuda.FloatTensor
                
        x = make_input(device, batch_size = 10, N_atoms = 5)
                
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        print('----------------------------------------------------------------')
        line_new = '{:25}  {:25} {:15}'.format('Layer (type)', 'Output Shape', 'Param #')
        print(line_new)
        print('================================================================')
        total_params = 0
        trainable_params = 0
        
        for layer in summary:
            ## input_shape, output_shape, trainable, nb_params
            trainable_params_print = 0
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
                    trainable_params_print = summary[layer]['nb_params']
            string_aux = str(summary[layer]["output_shape"])
            line_new  = f'{layer:{25}}  '
            line_new += f'{string_aux:{25}} '
            line_new += f'{trainable_params_print:{15}}'
            total_params += summary[layer]['nb_params']
            print(line_new)
        print('================================================================')
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str(total_params - trainable_params))
        print('----------------------------------------------------------------')
        return summary