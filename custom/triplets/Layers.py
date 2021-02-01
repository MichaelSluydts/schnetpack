import torch.nn as nn
from schnetpack2.data import Structure

class CNN_embeddings(nn.Module):
    def __init__(self, model):
        super(CNN_embeddings, self).__init__()
        self.model = model

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        inputs[Structure.R].requires_grad_()
        inputs['representation'] = self.model.representation(inputs)
        
        atomic_numbers = inputs[Structure.Z]
        atom_mask = inputs[Structure.atom_mask]
        
        y = self.model.output_modules.unwrapper(inputs)
#        y = self.out_net1(y)

        if self.model.output_modules.atomref is not None:
            y0 = self.model.output_modules.atomref(atomic_numbers)
            yi = y + y0

        y = self.model.output_modules.atom_pool(yi, atom_mask)

        y = self.model.output_modules.out_net2(y)

        result = {"y": y, "natoms" : inputs['_atomic_numbers']}

        return result


