import torch,pdb
import pdb

def gaussian_NLL(y_true, mu, sigma):
    return torch.mean(torch.log(sigma) + 0.5*(y_true - mu).pow(2)/(sigma.pow(2)),dim=-1) + 1e-6

def gaussian_NLL_forces(y_true, mu, sigma):
    return torch.log(sigma+1e-4) + 0.5*(y_true - mu).pow(2)/(sigma.pow(2)+1e-4) + 1e-5

#def gaussian_NLL_mod(y_true, mu, sigma):
#    return torch.mean(torch.log(sigma+1e-2) + 0.5*(y_true - mu).pow(2)/(sigma+1e-2).pow(2),dim=-1) + 1e-6
#
#def gaussian_NLL_forces_mod(y_true, mu, sigma):
#    return torch.log(sigma+1e-2) + 0.5*(y_true - mu).pow(2)/(sigma+1e-2).pow(2) + 1e-6

def gaussian_NLL_mod(y_true, mu, sigma):    
    frac_log = torch.log((y_true - mu).abs()+1e-5)-torch.log(sigma+1e-5)
    frac     = torch.exp(2*frac_log)
    
    gaussian_NLL = torch.log(sigma+1e-5) + 0.5*frac

    return torch.mean(gaussian_NLL, dim=-1)
    
def gaussian_NLL_forces_mod(y_true, mu, sigma):
    frac_log = torch.log((y_true - mu).abs()+1e-5)-torch.log(sigma+1e-2)
    frac     = torch.exp(2*frac_log)
    
    gaussian_NLL = torch.log(sigma+1e-2) + 0.5*frac

    return gaussian_NLL
    
#def gaussian_NLL_mod(y_true, mu, sigma):
##    mask = 10*sigma<(y_true-mu).abs()#torch.nn.functional.relu6((sigma/10-y_true-mu).abs())/6
##    mse  = (y_true-mu).pow(2)
#    
#    frac_log = torch.log((y_true - mu).abs()+1e-5)-torch.log(sigma+1e-5)
#    frac     = torch.exp(2*frac_log)
#    
#    gaussian_NLL = torch.log(sigma+1e-5) + 0.5*frac
#
#    return torch.mean(gaussian_NLL, dim=-1)    
##    return torch.mean(mask.float()*mse+(~mask).float()*gaussian_NLL,dim=-1)
##    return torch.mean(mask*mse+(1-mask)*gaussian_NLL,dim=-1)
##    return torch.mean(gaussian_NLL, dim=-1)
#
#def gaussian_NLL_forces_mod(y_true, mu, sigma):
##    mask = 10*sigma<(y_true-mu).abs()#torch.nn.functional.relu6((sigma/10-y_true-mu).abs())/6
##    mse  = (y_true-mu).pow(2)
#    
#    frac_log = torch.log((y_true - mu).abs()+1e-5)-torch.log(sigma+1e-5)
#    frac     = torch.exp(2*frac_log)
#    
#    gaussian_NLL = torch.log(sigma+1e-5) + 0.5*frac
#    
#    return gaussian_NLL
#    return mask.float()*mse+(~mask).float()*gaussian_NLL
#    return torch.mean(mask*mse+(1-mask)*gaussian_NLL,dim=-1)

def logrootloss(result, batch,kef=100,kf=1,ke=1):
    N = 3 * torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)

    ediff = batch['energy'].type(result["y"].dtype) - result['y']
    ediff = ediff ** 2
    emean = torch.mean(3 * ediff / N.t())

    if 'dydx' in results.keys():
        fdiff = batch['forces'].type(result["y"].dtype) - result['dydx']
        fdiff = fdiff ** 2
        fmean = torch.mean(torch.sum(fdiff, [1, 2]) / N.t())
    else:
        fmean = 0.
    #diff = 0.9*ediff.mean() + 0.1*fdiff.mean()
#        err_sq = torch.mean(ediff)



    diff= torch.pow(torch.log(1+kef*emean*fmean+kf*fmean+ke*emean),0.5)
    return diff


def MSEloss(result, batch,kf=0.001,ke=1,kef=0,mean=0.,stddev=1.,property='energy', standardize=False, per_atom=True):
    #if per_atom:
    N = torch.tensor([[3.]]).type(result["y"].dtype).cuda()
    #else:
    #    N = 3 * torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)
    Nf = 3. * torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)
    if standardize:
        ediff = ((batch[property] - mean.cuda())/stddev.cuda()).type(result["y"].dtype) - result['y']
    else:
        ediff = batch[property].type(result["y"].dtype) - result['y']
    ediff = ediff ** 2
    emean = 3. * torch.mean(ediff / N.t())

    if 'dydx' in result.keys():
        fdiff = batch['forces'].type(result["dydx"].dtype) - result['dydx']
        fdiff = fdiff ** 2
        fmean = torch.mean(torch.sum(fdiff,[1,2])/Nf.t())
    else:
        fmean = 0.
    diff = kf*fmean + ke*emean+kef*fmean*emean
    return diff

def MAEloss(result, batch,kf=0.001,ke=1,kef=0, property='energy',mean=0.,stddev=1., standardize=False, per_atom = True):
    #if not per_atom:
    N = torch.tensor([[3.]]).type(result["y"].dtype).cuda()
    #else:
    #    N = 3 * torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)
    Nf = 3. * torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)

    if standardize:
        ediff = torch.abs(((batch[property] - mean.cuda()) / stddev.cuda()).type(result["y"].dtype) - result['y'])
    else:
        ediff = torch.abs(batch[property].type(result["y"].dtype) - result['y'])
    emean = 3. * torch.mean(ediff / N.t())

    if 'dydx' in result.keys():
        fdiff = torch.abs(batch['forces'].type(result["dydx"].dtype) - result['dydx'])
        fmean = torch.mean(torch.sum(fdiff, [1, 2]) / Nf.t())
    else:
        fmean = 0.
    diff = kf * fmean + ke * emean + kef * fmean * emean
    return diff

def NLLMSEloss(result, batch,mean=None, std=None, std_forces=None,kf=0.001,ke=1,kef=0, kgamma=0.0):
#    if mean is not None and std is not None: batch['energy'] = (batch['energy']-mean)/std
#    if mean is not None and std is not None: result["dydx"]  = result["dydx"]*std
#    pdb.set_trace()
    result["y"]            = (result["y"]-mean)/std
#    result["dydx"]         = result["dydx"]/std_forces
    result['sigma']        = result['sigma']/std#+1e-3
#    result['sigma_forces'] = result['sigma_forces']*std
    
    batch['energy'] = (batch['energy']-mean)/std
#    batch['forces'] = batch['forces']/std_forces
    
    ediff = gaussian_NLL(batch['energy'].type(result["y"].dtype), result['y'], result['sigma'])
    fdiff = batch['forces'].type(result["y"].dtype) - result['dydx']
    fdiff = fdiff ** 2
    N= 3*torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)
    fmean = torch.mean(torch.sum(fdiff,[1,2])/N.t())
    emean = 3*torch.mean(ediff/N.t())
    diff = kf*fmean + ke*emean + kef*fmean*emean + kgamma*result["gamma"].pow(2).sum()
    
    result["y"]            = result["y"]*std+mean
#    result["dydx"]         = result["dydx"]*std_forces
    result['sigma']        = result['sigma']*std
#    result['sigma_forces'] = result['sigma_forces']*std_forces
    
    batch['energy'] = batch['energy']*std + mean
#    batch['forces'] = batch['forces']*std_forces
#    if mean is not None and std is not None: result["dydx"]  = result["dydx"]/std
    if diff!=diff: raise ValueError("Loss has become NaN exiting calculation")
    return diff
    
def NLLMSEloss_forces(result, batch,mean=None, std=None, mean_forces=None, std_forces=None,kf=0.00001,ke=1,kef=0):
#    if mean is not None and std is not None: batch['energy'] = (batch['energy']-mean)/std
#    if mean is not None and std is not None: batch['forces'] = batch['forces']/std
    result["y"]            = (result["y"]-mean)/std
    result["dydx"]         = (result["dydx"]-mean_forces)/std_forces
#    result['sigma']        = result['sigma']/std
#    result['sigma_forces'] = result['sigma_forces']*std
    
    batch['energy'] = (batch['energy']-mean)/std
    batch['forces'] = (batch['forces']-mean_forces)/std_forces
    
    ediff = gaussian_NLL_mod(batch['energy'].type(result["y"].dtype), result['y'], result['sigma'])
    fdiff = gaussian_NLL_forces_mod(batch['forces'].type(result["dydx"].dtype), result['dydx'], result['sigma_forces'])
    N= 3*torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)
    fmean = torch.mean(torch.sum(fdiff,[1,2])/N.t())
    emean = 3*torch.mean(ediff/N.t())
    
    fdiff = batch['forces'].type(result["y"].dtype) - result['dydx']
    fdiff = fdiff ** 2
    fmean += 200*torch.mean(torch.sum(fdiff,[1,2])/N.t())

    result["y"]            = result["y"]*std+mean
    result["dydx"]         = result["dydx"]*std_forces + mean_forces
    result['sigma']        = result['sigma']*std
    result['sigma_forces'] = result['sigma_forces']*std_forces
    
    batch['energy'] = batch['energy']*std + mean
    batch['forces'] = batch['forces']*std_forces + mean_forces
    
#    fmean = torch.exp(fmean)
#    emean = torch.exp(emean)
#    fmean = torch.nn.functional.softplus(fmean)
#    emean = torch.nn.functional.softplus(emean)
#    emean = torch.nn.functional.leaky_relu(emean, negative_slope=0.01)
#    fmean = torch.nn.functional.leaky_relu(fmean, negative_slope=0.01)
#    conv_factor = emean.data.abs().mean()/fmean.data.abs().mean()
#    conv_factor = conv_factor.detach()
    diff = kf*fmean + ke*emean+kef*fmean*emean
    return diff
    
def PhysNetLoss(result, batch, we = 1.0, wf = 0.0, wp = 0.0, wq = 0.0, wc=0.0, l_nh = 0.0):
    ediff  = batch['energy'].type(result["y"].dtype) - result['y']
    ediff  = ediff.abs()
    fdiff  = batch['forces'].type(result["y"].dtype) - result['dydx']
    fdiff  = fdiff.abs()
    dpdiff = batch['dipole'].type(result["y"].dtype) - result['dipole']
    dpdiff = dpdiff.abs()
    qdiff  = 0 - result['Q']
    qdiff  = qdiff.abs()
    mask   = ((batch['atom_index_0']+batch['atom_index_1'])>0)
    cdiff  = batch['scalar_coupling_constant'][mask].type(result["y"].dtype) - result['coupling_constant'][mask]
    cdiff  = cdiff.abs()
    N= 3*torch.sum((result['natoms'] != 0).type(result["y"].dtype), 1, keepdim=True)
    emean  = torch.mean(ediff)    
    fmean  = torch.mean(torch.sum(fdiff,[1,2])/N.t()/3)
    dpmean = torch.mean(dpdiff)
    qmean  = torch.mean(qdiff)
    cmean  = torch.mean(cdiff)
    
    mask = result["atom_mask"][:,1:,:]
    
    nh_loss = torch.mean(torch.sum(result['Ei'][:,1:].abs()*mask/(result['Ei'][:,:-1].abs()+result['Ei'][:,1:].abs()+1e-5),[1,2])/N.t())
    nh_loss = nh_loss + torch.mean(torch.sum(result['qi'][:,1:].abs()*mask/(result['qi'][:,:-1].abs()+result['qi'][:,1:].abs()+1e-5),[1,2])/N.t())
    
    return we*emean + wf*fmean + wp*dpmean + wq*qmean + wc*cmean + l_nh/2*nh_loss  
