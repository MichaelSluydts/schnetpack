from torch.nn.modules import Module
import torch.nn.functional as F
import torch.nn as nn

from fastai.layers import CrossEntropyFlat
from schnetpack2.custom.triplets.Losses_triplets import SemihardNegativeTripletSelector, WeightedNegativeTripletSelector, semihard_negative

import pdb

class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, result, batch):
        return self.loss(result["y"],batch["_idx"].squeeze())
    

class TripletLoss(Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.__name__ = "TripletLoss"

    def forward(self, result, batch):
        anchor, positive, negative = result["y"].view(3,result["y"].shape[0]//3,result["y"].shape[1])
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() #if size_average else losses.sum()
        
def TripletLoss_val(result, batch, margin=1.0):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean() #if size_average else losses.sum()

class OnlineTripletLoss(Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.name_loss = "OnlineTripletLoss"
        self.__name__ =  triplet_selector.negative_selection_fn.__name__
        self.last_loss = None

    def forward(self, result, batch):
        embeddings = result["y"]
        labels     = batch["_idx"].squeeze()
        
        triplets = self.triplet_selector.get_triplets(embeddings, labels)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)
        self.last_loss = losses.mean()
        
        return losses.mean()#, len(triplets)
        
    def name(self):
        return self.name_loss
        
    def value(self):
        return float(self.last_loss)

def masked_RMSE_loss(output, target_reg, scale_factors):
    target_mask = ~(target_reg.data!=target_reg.data)
    loss = sum([scale_factor*((output[:,col][target_mask[:,col]] - target_reg[:,col][target_mask[:,col]])**2).sum(0) for col, scale_factor in enumerate(scale_factors)])
    return (loss/output.nelement()+1e-7).sqrt()

def masked_MSE_loss(output, target_reg, scale_factors):
    target_mask = ~(target_reg.data!=target_reg.data)
    loss = sum([scale_factor*((output[:,col][target_mask[:,col]] - target_reg[:,col][target_mask[:,col]])**2).sum(0) for col, scale_factor in enumerate(scale_factors)])
    return loss/output.nelement()#(loss.sum(0)*scale_factors).sum()/output.nelement()
#    return F.mse_loss(output[target_mask], target_reg[target_mask], reduction='mean')#size_average=True, reduce=True)