import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
#import torchdyn
import torch
import numpy as np
import time
import pprint
import matplotlib.pyplot as plt
import normflows as nf
import utils
from sklearn.decomposition import PCA
#from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
#from torch import nn
from tqdm import tqdm
from torchdyn.core import NeuralODE
#from torchdyn.datasets import generate_moons
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from torchcfm.conditional_flow_matching import *
from torchcfm.models.models import *
from torchcfm.utils import *

def preprocess_samples(filepath, split, batch_size, dataset_limiter):
    
    #with open(filepath, 'r') as f:
        #rawdata = np.loadtxt(f)
    
    rawdata = np.load(filepath)

    rawdata = rawdata[:dataset_limiter]

    alldata = torch.from_numpy(rawdata)

    tensor_dataset = torch.utils.data.TensorDataset(alldata)

    train, valid, test = random_split(tensor_dataset, split, generator=torch.Generator().manual_seed(42))

    train_np = train[:][0].numpy()
    valid_np = valid[:][0].numpy()
    test_np = test[:][0].numpy()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader, train_np, valid_np, test_np

def init_model(dimensionality, w, sigma, model_style):

  model = MLP(dim=dimensionality, w=w, time_varying=True)
  #optimizer = torch.optim.Adam(model.parameters())
  if model_style == 'ConditionalFlowMatcher':
    FM = ConditionalFlowMatcher(sigma=sigma)
  elif model_style == 'ExactOptimalTransportConditionalFlowMatcher':
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
  else:
    FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="exact")

  base = nf.distributions.base.DiagGaussian(dimensionality)

  return model, FM, base

def train_epoch(loader, model, FM, base, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    loss_hist = np.ndarray([])

    #for x in tqdm(loader):
    for x in loader:

      x = x[0]
      x = x.float()

      optimizer.zero_grad()

      x0 = base.sample(len(x))
      t, xt, ut = FM.sample_location_and_conditional_flow(x0, x)

      vt = model(torch.cat([xt, t[:, None]], dim=-1))
      loss = torch.mean((vt - ut) ** 2)

      loss.backward()
      optimizer.step()

      loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

def sample(model, base, num_samples):
  node = NeuralODE(torch_wrapper(model), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)
  with torch.no_grad():
    traj = node.trajectory(base.sample(num_samples),t_span=torch.linspace(0, 1, 100),)
  #plot_trajectories(traj.cpu().numpy())
  samples = traj[-1]
  return samples

def sampler(model, base, samples):

  model.eval()
  start = time.time()
  gendata = sample(model, base, samples).numpy()
  pass
  end = time.time()
  delta = end - start
  #print(f"Generated {samples} samples in {delta} seconds at a rate of {samples/delta} samples per second.")
  return gendata, samples/delta

def train_and_sample(model, FM, base, loader, valid_np, test_np, pca, bounds, lr):

  valid_KLD_hist = np.array([])
  valid_WD_hist = np.array([])

  while valid_KLD_hist.shape[0] < 5 or np.mean(valid_KLD_hist[-3:] - valid_KLD_hist[-4:-1]) < 0:

    train_epoch(loader, model, FM, base, lr)

    model.eval()

    generated_samples = sampler(model, base, len(valid_np))[0]
    valid_KLD = utils.counts_to_KLD(generated_samples, valid_np, pca, bounds)
    valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
    valid_WD = utils.counts_to_WD(generated_samples, valid_np, pca, bounds)
    valid_WD_hist = np.append(valid_WD_hist, valid_WD)
    #print('KLD history:', valid_KLD_hist)
    #print('WD history:', valid_WD_hist)

  generated_testing, speed = sampler(model, base, len(test_np))
  final_KLD_score = utils.counts_to_KLD(generated_testing, test_np, pca, bounds)
  final_WD_score = utils.counts_to_WD(generated_testing, test_np, pca, bounds)

  info = {'Architecture': 'CNF', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': len(valid_KLD_hist),
          'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0])}

  return generated_testing, info