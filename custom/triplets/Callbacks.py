from itertools import combinations
import numpy as np

import torch
from torch.autograd import Variable

from fastai.callback import AverageMetric, Callback
from fastai.core import is_listy

from schnetpack2.custom.triplets.Losses_triplets import pdist

import pdb

class AverageMetricNaNs(AverageMetric):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, func):
        super(AverageMetricNaNs, self).__init__(func)

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        met_val = self.func(last_output, *last_target)
        if met_val is not None:
            self.count += last_target[0].size(0)
            self.val   += last_target[0].size(0) * met_val.detach().cpu()
            
class TripletSetter(Callback):
    def __init__(self, model, dl, dl2, negative_selection_fn, margin=1.0, triplets_per_class=500):
        self.model = model
        self.dl  = dl
        self.dl2 = dl2
        self.triplets_per_class = triplets_per_class
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

        
    def on_epoch_begin(self, **kwargs):
        self.dl.dataset._set_triplets(self.get_triplets())
        self.dl.dataset.transform.reset()
        
    def get_triplets(self):
        
        embeddings, labels = self.get_embeddings()
        
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        triplets = []
        
        labels = labels.flatten()

        for ind, label in enumerate(set(labels)):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values, self.margin)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])
                if len(triplets)>=(ind+1)*self.triplets_per_class: 
                    triplets = triplets[:(ind+1)*self.triplets_per_class]
                    break
                    
        return triplets
    
    def get_embeddings(self):
        dataiter = iter(self.dl2)
    
        embeddings = []
        targets    = []
        
        for ind, atoms in enumerate(dataiter):
            atoms = {k:v.cuda() for k,v in atoms.items()}
            embedding = self.model(atoms) #TODO: check if is already cuda tensor at this point

            targets.extend(atoms['_idx'].cpu())#atoms["labels"].cpu())                
            embeddings.append(embedding["y"].detach().cpu())
            
        return torch.cat(embeddings, 0), torch.cat(targets, 0).numpy()