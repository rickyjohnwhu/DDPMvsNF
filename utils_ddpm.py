import pprint
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch.nn as nn
import utils
from functorch import vmap
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torch import nn
from sklearn.decomposition import PCA
from Unet1D import Unet1D
from backbone import ConvBackbone1D
from tqdm import tqdm

class number_dataset(Dataset):

    def __init__(self, datatensor):
      datatensor = datatensor.unsqueeze(1)
      self.data = datatensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        return x.float()

def preprocess_samples(filepath, split, batch_size, dataset_limiter):
    
    rawdata = np.load(filepath)

    rawdata = rawdata[:dataset_limiter]

    alldata = torch.from_numpy(rawdata).to('cpu')

    num_set = number_dataset(alldata)

    train, valid, test = random_split(num_set, split)

    train_np = train[:].squeeze(1).to('cpu').numpy()
    valid_np = valid[:].squeeze(1).to('cpu').numpy()
    test_np = test[:].squeeze(1).to('cpu').numpy()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader, train_np, valid_np, test_np, num_set

def polynomial_noise(t, alpha_max, alpha_min, s=1e-5):

    T = t[-1]
    alphas = (1-2*s)*(1-(t/T)**2) + s
    a = alphas[1:]/alphas[:-1]
    a[a**2 < 0.001] = 0.001
    alpha_schedule = torch.cumprod(a, 0)
    return alpha_schedule

NOISE_FUNCS = {
    "polynomial": polynomial_noise,
              }

class VPDiffusion:
    
    def __init__(self,
                 num_diffusion_timesteps,
                 noise_schedule="polynomial",
                 alpha_max=20.,
                 alpha_min=0.01,
                 NOISE_FUNCS=NOISE_FUNCS,
                ):

        self.bmul = vmap(torch.mul)
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.times = torch.arange(num_diffusion_timesteps)
        self.alphas = NOISE_FUNCS[noise_schedule](torch.arange(num_diffusion_timesteps+1), alpha_max, alpha_min)

    def get_alphas(self):
        return self.alphas

    def forward_kernel(self, x0, t):

        alphas_t = self.alphas[t]
        noise = torch.randn_like(x0)
        x_t = self.bmul(x0, alphas_t.sqrt()) + self.bmul(noise, (1-alphas_t).sqrt())
        return x_t, noise

    def reverse_kernel(self, x_t, t, backbone, pred_type):
        
        alphas_t = self.alphas[t]

        if pred_type == "noise":
            noise = backbone(x_t, alphas_t)
            noise_interp = self.bmul(noise, (1-alphas_t).sqrt())
            x0_t = self.bmul((x_t - noise_interp), 1/alphas_t.sqrt())
            
        elif pred_type == "x0":
            x0_t = backbone(x_t, alphas_t)
            x0_interp = self.bmul(x0_t, (alphas_t).sqrt())
            noise = self.bmul((x_t - x0_interp), 1/(1-alphas_t).sqrt())
        
        else:
            raise Exception("Please provide a valid prediction type: 'noise' or 'x0'")

        return x0_t, noise

    def reverse_step(self, x_t, t, t_next, backbone, pred_type):

        alphas_t = self.alphas[t]
        alphas_t_next = self.alphas[t_next]
        sigmas_t = ((1 - alphas_t_next) / (1- alphas_t)).sqrt() * (1 - alphas_t / alphas_t_next).sqrt()

        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type)

        output_shape = x0_t.size()
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul((1 - alphas_t_next - (sigmas_t**2)).sqrt(), noise) + self.bmul(sigmas_t, torch.randn(output_shape))
        return xt_next

    def sample_prior(self, xt):
        
        noise = torch.randn_like(xt)
        return noise

def init_model_backbone(loader, resnet_block_groups, learned_sinusoidal_cond, learned_sinusoidal_dim, dim_mults, num_set, lr, model_dim):

    num_torsions = loader.dataset[0].shape[-1]
    
    model = Unet1D(dim=model_dim, channels=1, resnet_block_groups=resnet_block_groups, learned_sinusoidal_cond=learned_sinusoidal_cond, learned_sinusoidal_dim=learned_sinusoidal_dim, dim_mults=dim_mults)

    backbone = ConvBackbone1D(model=model, data_shape=num_torsions, target_shape=model_dim, num_dims=len(num_set.data.shape), lr=lr)
    return model, backbone, num_torsions

def l2_loss(x, x_pred):
    return (x - x_pred).pow(2).sum((1,2)).pow(0.5).mean()

def train_epoch(loader, backbone, diffusion):

    for b in loader:

        t = torch.randint(low=0,high=diffusion.num_diffusion_timesteps,size=(b.size(0),)).long()
      
        b_t, e_0 = diffusion.forward_kernel(b, t)
        b_0, e_t = diffusion.reverse_kernel(b_t, t, backbone, "x0")

        loss = l2_loss(b, b_0)
        backbone.optim.zero_grad()
        loss.backward()
        backbone.optim.step()

def sample_batch(batch_size, loader, diffusion, backbone, num_set, pred_type="x0"):

    def sample_prior(batch_size, shape):
        
        prior_sample = torch.randn(batch_size, *shape[1:], dtype=torch.float)
        return prior_sample

    def get_adjacent_times(times):
        
        times_next = torch.cat((torch.Tensor([0]).long(), times[:-1]))
        return list(zip(reversed(times), reversed(times_next)))

    xt = sample_prior(batch_size, num_set.data.shape)
    time_pairs = get_adjacent_times(diffusion.times)

    for t, t_next in time_pairs:
        t = torch.Tensor.repeat(t, batch_size)
        t_next = torch.Tensor.repeat(t_next, batch_size)
        xt_next = diffusion.reverse_step(xt, t, t_next, backbone, pred_type=pred_type)
        xt = xt_next
    return xt

def sample_loop(num_samples, batch_size, loader, diffusion, backbone, num_torsions, num_set):

    gendata = torch.empty(0, 1, num_torsions)

    n_runs = max(num_samples//batch_size, 1)
    if num_samples <= batch_size:
        batch_size = num_samples

    with torch.no_grad():
        for save_idx in range(n_runs):
            x0 = sample_batch(batch_size, loader, diffusion, backbone, num_set)
            gendata = torch.cat((gendata, x0), 0)

    return gendata

def sampler(samples, batchsize, loader, diffusion, backbone, num_torsions, num_set):
    
    start = time.time()
    gendata = sample_loop(samples, batchsize, loader, diffusion, backbone, num_torsions, num_set)
    pass
    end = time.time()
    delta = end - start
    #print(f"Generated {samples} samples in {delta} seconds at a rate of {samples/delta} samples per second.")
    gendata = gendata.squeeze(1)
    return gendata, samples/delta

def train_and_sample(model, loader, valid_np, test_np, train_np, pca, bounds, diffusion, backbone, num_torsions, num_set, sample_batching, epoch_max, aib9_status):

    #valid_KLD_hist = np.array([])
    #valid_WD_hist = np.array([])

    fullstart = time.time()
    
    #while valid_KLD_hist.shape[0] < 5 or np.mean(valid_KLD_hist[-3:] - valid_KLD_hist[-4:-1]) < 0:
    epochs = 0
    while epochs < epoch_max:

        train_epoch(loader, backbone, diffusion)

        epochs += 1

        #model.eval()

        #generated_samples = sampler(len(valid_np), sample_batching, loader, diffusion, backbone, num_torsions, num_set)[0]

        #valid_KLD = utils.counts_to_KLD(generated_samples, valid_np, pca, bounds)
        #valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
        #valid_WD = utils.counts_to_WD(generated_samples, valid_np, pca, bounds)
        #valid_WD_hist = np.append(valid_WD_hist, valid_WD)

        #print('KLD history:', valid_KLD_hist)
        #print('WD history:', valid_WD_hist)

    model.eval()
    generated_testing, speed = sampler(len(test_np), sample_batching, loader, diffusion, backbone, num_torsions, num_set)
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
        info = {'Architecture': 'DDPM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart, 'Residue KLD': res_KLD_array}
    else:
        info = {'Architecture': 'DDPM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart}

    return generated_testing, info