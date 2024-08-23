import math
import os
import time
import numpy as np
import ot as pot
import torch
import torch.nn as nn
import normflows as nf
import utils
from sklearn.decomposition import PCA
from torchdyn.core import NeuralODE
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *
from Unet1D import Unet1D
import torch.nn.functional as F

class Interpolater:
    
    def __init__(self,
                 data_shape: tuple,
                 target_shape: tuple):
        self.data_shape, self.target_shape = data_shape, target_shape

    def to_target(self, x):
        return F.interpolate(x, size=self.target_shape, mode='nearest-exact')

    def from_target(self, x):
        return F.interpolate(x, size=self.data_shape, mode='nearest-exact')

def preprocess_samples(filepath, split, batch_size, dataset_limiter, device):
    
    rawdata = np.load(filepath)
    rawdata = rawdata[:dataset_limiter]

    alldata = torch.from_numpy(rawdata).to(device)
    tensor_dataset = torch.utils.data.TensorDataset(alldata)

    train, valid, test = random_split(tensor_dataset, split)

    train_np = train[:][0].to('cpu').numpy()
    valid_np = valid[:][0].to('cpu').numpy()
    test_np = test[:][0].to('cpu').numpy()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    
    return train_loader, train_np, valid_np, test_np

def init_model(dimensionality, w, sigma, model_style, time_varying, resnet_block_groups, learned_sinusoidal_cond, learned_sinusoidal_dim, dim_mults, model_dim, device):
    
    #model = MLP(dim=dimensionality, w=w, time_varying=time_varying)
    model = Unet1D(dim=model_dim, channels=1, resnet_block_groups=resnet_block_groups, learned_sinusoidal_cond=learned_sinusoidal_cond, learned_sinusoidal_dim=learned_sinusoidal_dim, dim_mults=dim_mults)

    if model_style == 'ConditionalFlowMatcher':
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif model_style == 'ExactOptimalTransportConditionalFlowMatcher':
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif model_style == 'SchrodingerBridgeConditionalFlowMatcher':
        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="exact")
    else:
        FM = TargetConditionalFlowMatcher(sigma=sigma)

    model.to(device)
    base = nf.distributions.base.DiagGaussian(dimensionality).to(device)
    
    return model, FM, base

def train_epoch(loader, model, FM, base, lr, interp):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_hist = np.ndarray([])

    for x in loader:
        
        x = x[0]
        x = x.float()
        
        optimizer.zero_grad()
        x0 = base.sample(len(x))
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x)
        
        xt = xt.unsqueeze(1)
        xt_up = interp.to_target(xt)
        vt_up = model(xt_up, t)
        vt = interp.from_target(vt_up)
        vt = vt.squeeze(1)
      
        #vt = model(torch.cat([xt, t[:, None]], dim=-1))
      
        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

def sample(model, base, num_samples, interp):
    
    #node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    
    #with torch.no_grad():
        #traj = node.trajectory(base.sample(num_samples),t_span=torch.linspace(0, 1, 100),)
    #samples = traj[-1]

    #return samples

    t_span=torch.linspace(0, 1, 100)
    dt = (t_span[-1] - t_span[0])/(len(t_span) - 1)

    xt = base.sample(num_samples)

    with torch.no_grad():
        
        for t in t_span:
            
            xt = xt.unsqueeze(1)
            xt_up = interp.to_target(xt)
        
            vt_up = model(xt_up, torch.full((xt.shape[0],), t).to('cuda:0'))
            vt = interp.from_target(vt_up)
            
            xt = xt.squeeze(1)
            vt = vt.squeeze(1)

            xt += vt*dt

    return xt
    
def sampler(model, base, samples, interp):
    
    model.eval()
    start = time.time()
    #gendata = sample(model, base, samples, interp).to('cpu').numpy()
    gendata = sample(model, base, samples, interp).detach().to('cpu').numpy()
    pass
    end = time.time()
    delta = end - start
    
    return gendata, samples/delta

def train_and_sample(model, FM, base, interp, loader, valid_np, test_np, train_np, pca, bounds, lr, epoch_max, aib9_status):
    
    #valid_KLD_hist = np.array([])
    #valid_WD_hist = np.array([])
    
    fullstart = time.time()
    
    #while valid_KLD_hist.shape[0] < 5 or np.mean(valid_KLD_hist[-3:] - valid_KLD_hist[-4:-1]) < 0:
    epochs = 0
    while epochs < epoch_max:
        
        train_epoch(loader, model, FM, base, lr, interp)

        epochs += 1

        #model.eval()

        #generated_samples = sampler(model, base, len(valid_np))[0]
        #valid_KLD = utils.counts_to_KLD(generated_samples, valid_np, pca, bounds)
        #valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
        #valid_WD = utils.counts_to_WD(generated_samples, valid_np, pca, bounds)
        #valid_WD_hist = np.append(valid_WD_hist, valid_WD)
        #print('KLD history:', valid_KLD_hist)
        #print('WD history:', valid_WD_hist)

    model.eval()
    generated_testing, speed = sampler(model, base, len(test_np), interp)

    aib9_status = True
    
    if aib9_status:
        res_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        res_KLD_array = np.array([])
        for res in res_array:
            KLD_pca = PCA(n_components=2)
            training_KLD_pca = KLD_pca.fit_transform(train_np[:, int(res*2):int((res+1)*2)])
            res_KLD_score = utils.counts_to_KLD(generated_testing[:, int(res*2):int((res+1)*2)], test_np[:, int(res*2):int((res+1)*2)], KLD_pca, bounds)
            res_KLD_array = np.append(res_KLD_array, res_KLD_score)
    
    final_KLD_score = utils.counts_to_KLD(generated_testing, test_np, pca, bounds)
    final_WD_score = utils.counts_to_WD(generated_testing, test_np, pca, bounds)
    #iterations = len(valid_KLD_hist)
    iterations = epochs

    fullstop = time.time()

    if aib9_status:
        info = {'Architecture': 'CFM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart, 'Residue KLD': res_KLD_array}
    else:
        info = {'Architecture': 'CFM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart}

    return generated_testing, info
