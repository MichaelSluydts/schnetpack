import numpy as np
import torch
device = torch.device("cuda")

def save_embeddings_triplets(data_loader, model, savename):
    
    data_iter = iter(data_loader)
    model.to(device)
    
    embeddings = []
    labels     = []
    
    for ind, input in enumerate(data_iter):
        input = {k:v.to(device) for k,v in input.items()}
        
        result = model(input)
        
        labels.append(input["_idx"].detach().cpu().numpy())
        embeddings.append(result["y"].detach().cpu().numpy())
    
    result = {}
    
    result["embeddings"] = np.vstack(embeddings)
    result["labels"] = np.vstack(labels).squeeze()
    
    np.save(savename, result)
    
