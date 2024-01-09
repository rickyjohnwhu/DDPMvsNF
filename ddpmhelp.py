import pprint
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import torch.nn as nn
import 
from functorch import vmap
from scipy.stats import wasserstein_distance
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

  with open(filepath, 'r') as f:
    rawdata = np.loadtxt(f)

    rawdata = rawdata[:dataset_limiter]

    alldata = torch.from_numpy(rawdata)

    num_set = number_dataset(alldata)

    train, valid, test = random_split(num_set, split, generator=torch.Generator().manual_seed(42))

    train_np = train[:].squeeze(1).numpy()
    valid_np = valid[:].squeeze(1).numpy()
    test_np = test[:].squeeze(1).numpy()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader, train_np, valid_np, test_np, num_set

def polynomial_noise(t, alpha_max, alpha_min, s=1e-5):
    """
    Same schedule used in Hoogeboom et. al. (Equivariant Diffusion for Molecule Generation in 3D)
    """
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
    """
    Performs a diffusion according to the VP-SDE.
    """
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
        self.alphas = NOISE_FUNCS[noise_schedule](torch.arange(num_diffusion_timesteps+1),
                                                  alpha_max,
                                                  alpha_min)

    def get_alphas(self):
        return self.alphas

    def forward_kernel(self, x0, t):
        """
        Maginal transtion kernels of the forward process. p(x_t|x_0).
        """
        alphas_t = self.alphas[t]
        noise = torch.randn_like(x0)
        # interpolate between data and noise
        x_t = self.bmul(x0, alphas_t.sqrt()) + self.bmul(noise, (1-alphas_t).sqrt())
        return x_t, noise

    def reverse_kernel(self, x_t, t, backbone, pred_type):
        """
        Marginal transition kernels of the reverse process. q(x_0|x_t).
        """
        # get noise schedule
        alphas_t = self.alphas[t]

        # predict noise added to data
        if pred_type == "noise":
            noise = backbone(x_t, alphas_t)
            noise_interp = self.bmul(noise, (1-alphas_t).sqrt())
            # predict x0 given x_t and noise
            x0_t = self.bmul((x_t - noise_interp), 1/alphas_t.sqrt())

        # predict data
        elif pred_type == "x0":
            x0_t = backbone(x_t, alphas_t)
            x0_interp = self.bmul(x0_t, (alphas_t).sqrt())
            # predict noise given x_t and x0
            noise = self.bmul((x_t - x0_interp), 1/(1-alphas_t).sqrt())
        else:
            raise Exception("Please provide a valid prediction type: 'noise' or 'x0'")

        return x0_t, noise

    def reverse_step(self, x_t, t, t_next, backbone, pred_type):
        """
        Stepwise transition kernel of the reverse process q(x_t-1|x_t).
        """

        # getting noise schedule
        alphas_t = self.alphas[t]
        alphas_t_next = self.alphas[t_next]
        sigmas_t = ((1 - alphas_t_next) / (1- alphas_t)).sqrt() * (1 - alphas_t / alphas_t_next).sqrt()

        # computing x_0' ~ p(x_0|x_t)
        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type)

        # computing x_t+1 = f(x_0', x_t, noise)
        output_shape = x0_t.size()
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul((1 - alphas_t_next - (sigmas_t**2)).sqrt(), noise) + self.bmul(sigmas_t, torch.randn(output_shape))
        return xt_next

    def sample_prior(self, xt):
        """
        Generates a sample from a prior distribution z ~ p(z).
        """
        noise = torch.randn_like(xt)
        return noise

def init_model_backbone(resnet_block_groups, learned_sinusoidal_dim, dim_mults, lr):

  num_torsions = train_np.__getitem__(0).shape[-1]
  model_dim = int(np.ceil(num_torsions/resnet_block_groups) * resnet_block_groups)

  #print(f"There are {num_torsions} torsion angles and {resnet_block_groups} resnet block groups,")
  #print(f"so the model should have {model_dim} dimensions")

  model = Unet1D(dim=model_dim,
               channels=1,
               resnet_block_groups=resnet_block_groups,
               learned_sinusoidal_cond=True,
               learned_sinusoidal_dim=learned_sinusoidal_dim,
               dim_mults=dim_mults
              )

  backbone = ConvBackbone1D(model=model, # model
                          data_shape=num_torsions, # data shape
                          target_shape=model_dim, # network shape
                          num_dims=len(num_set.data.shape),
                          lr=lr
                         )
  return model, backbone, num_torsions

def l2_loss(x, x_pred):
        return (x - x_pred).pow(2).sum((1,2)).pow(0.5).mean()

def train_epoch(loader, backbone, diffusion):

  #for b in tqdm(loader):
  for b in loader:

    # sample set of times
    t = torch.randint(low=0,high=diffusion.num_diffusion_timesteps,size=(b.size(0),)).long()

    # corrupt data according to noise schedule
    b_t, e_0 = diffusion.forward_kernel(b, t)

    # predict noise and original data
    b_0, e_t = diffusion.reverse_kernel(b_t, t, backbone, "x0")

    # evaluate loss and do backprop
    loss = l2_loss(b, b_0)
    backbone.optim.zero_grad()
    loss.backward()
    backbone.optim.step()

def sample_batch(batch_size, loader, diffusion, backbone, pred_type="x0"):

    def sample_prior(batch_size, shape):
        "Generates samples of gaussian noise"
        prior_sample = torch.randn(batch_size, *shape[1:], dtype=torch.float)
        return prior_sample

    def get_adjacent_times(times):
        """
        Pairs t with t+1 for all times in the time-discretization
        of the diffusion process.
        """
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

def sample_loop(num_samples, batch_size, loader, diffusion, backbone, num_torsions):

    gendata = torch.empty(0, 1, num_torsions)

    n_runs = max(num_samples//batch_size, 1)
    if num_samples <= batch_size:
        batch_size = num_samples

    with torch.no_grad():
        for save_idx in range(n_runs):
            x0 = sample_batch(batch_size, loader, diffusion, backbone)
            # print(f"Samples generated {(save_idx + 1) * batch_size}")
            gendata = torch.cat((gendata, x0), 0)

    return gendata

def sampler(samples, batchsize, loader, diffusion, backbone, num_torsions):
  start = time.time()
  gendata = sample_loop(samples, batchsize, loader, diffusion, backbone, num_torsions)
  pass
  end = time.time()
  delta = end - start
  #print(f"Generated {samples} samples in {delta} seconds at a rate of {samples/delta} samples per second.")
  gendata = gendata.squeeze(1)
  return gendata, samples/delta

def train_and_sample(model, loader, valid_np, test_np, pca, bounds, diffusion, backbone, num_torsions):

  valid_KLD_hist = np.array([])
  valid_WD_hist = np.array([])

  while valid_KLD_hist.shape[0] < 5 or np.mean(valid_KLD_hist[-3:] - valid_KLD_hist[-4:-1]) < 0:

    train_epoch(loader, backbone, diffusion)

    model.eval()

    generated_samples = sampler(len(valid_np), sample_batching, loader, diffusion, backbone, num_torsions)[0]

    valid_KLD = counts_to_KLD(generated_samples, valid_np, pca, bounds)
    valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
    valid_WD = counts_to_WD(generated_samples, valid_np, pca, bounds)
    valid_WD_hist = np.append(valid_WD_hist, valid_WD)

    #print('KLD history:', valid_KLD_hist)
    #print('WD history:', valid_WD_hist)

  generated_testing, speed = sampler(len(test_np), sample_batching, loader, diffusion, backbone, num_torsions)
  final_KLD_score = counts_to_KLD(generated_testing, test_np, pca, bounds)
  final_WD_score = counts_to_WD(generated_testing, test_np, pca, bounds)

  info = {'Architecture': 'DDPM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': count_parameters(model), 'Iterations': len(valid_KLD_hist),
          'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0])}

  return generated_testing, info