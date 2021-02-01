import torch,pdb

def Emetric(x,y,norm = True, mean=None, stddev=None,property='energy',standardize=False,per_atom=True):
#    if stddev is not None:
#        y['energy'] = y['energy']*stddev
#        x['y']      = x['y']*stddev

#    y['energy'] = y['energy']/100
#    x['y']      = x['y']/100 
    if standardize:
        diff = torch.abs(stddev.type(x["y"].dtype).cuda()*x['y']+mean.type(x["y"].dtype).cuda() - y[property].type(x["y"].dtype).cuda())
    else:
        diff = torch.abs(x['y'] - y[property].type(x["y"].dtype).cuda())
        #removed norm and did not further check standardize fix later
        #if per_atom:
    return diff.mean()
        #else:
        #    N= torch.sum((x['natoms'] != 0), 1, keepdim=True).type(x["y"].dtype)
        #    return torch.mean(diff/N.t())

def Fmetric(x,y, stddev=None):
#    if stddev is not None:
#        y['forces'] = y['forces']*stddev
#        x['dydx']   = x['dydx']*stddev
    #import pdb
#    y['forces'] = y['forces']/100
#    x['dydx']   = x['dydx']/100
    
    N= 3*torch.sum((x['natoms'] != 0), 1, keepdim=True).type(x["dydx"].dtype)

    diff = torch.abs(x['dydx'] - y['forces'].type(x["dydx"].dtype).cuda())
    return torch.mean(torch.sum(diff,[1,2])/N.t())
    
def DipoleMetric(result, batch):
    dpdiff = batch['dipole'].type(result["y"].dtype) - result['dipole']
    dpdiff = dpdiff.abs()
    dpmean = torch.mean(dpdiff)
    return dpmean

def ChargeMetric(result, batch):
    qdiff  = 0 - result['Q'].sum()
    qdiff  = qdiff.abs()
    qmean  = torch.mean(qdiff)
    return qmean
    
def CouplingMetric(result, batch):
    cdiff  = batch['scalar_coupling_constant'].type(result["y"].dtype) - result['coupling_constant']
    cdiff  = cdiff.abs()
    cmean  = torch.mean(cdiff)
    return cmean
    
def CouplingMetricKaggle(result, batch):
    targets = batch['scalar_coupling_constant'].type(result["y"].dtype).squeeze(-1)
    maes = []
    types_set = torch.unique(batch["type"])
    for t in types_set:
        mask = batch["type"].squeeze(-2)==t
        y_true = targets[mask]
        y_pred = result['coupling_constant'][mask]
        mae = ((y_true-y_pred).abs()+1e-9).mean().log()
        maes.append(mae)
    return torch.stack(maes).mean() 

def Debug(result,batch):
    pdb.set_trace()
    return 1
    
def uncertainty(x,y):
    return x['sigma'].abs().mean()#(x['sigma'].abs()/N.t()).mean()
    
def uncertainty_forces(x,y):
    N= 3*torch.sum((x['natoms'] != 0), 1, keepdim=True).type(x["dydx"].dtype)
    return torch.mean(torch.sum(x['sigma_forces'].abs(),[1,2])/N.t())


def accuracy(result, batch,prop,threshold=0.0):
    "Computes accuracy"
    #pdb.set_trace()
    sigmoid = torch.sigmoid(result['y'])
    predictions = (sigmoid > 0.5).type(sigmoid.dtype)
    targs = (batch[prop]>threshold).type(sigmoid.dtype)

    return (predictions==targs).float().mean()
