import torch
import numpy as np
import normflows as nf
import time
import matplotlib.pyplot as plt
import pprint
from utils import counts_to_KLD, counts_to_WD
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

def preprocess_samples(filepath, split, batch_size, device):

  with open(filepath, 'r') as f:
    rawdata = np.loadtxt(f)

  rawdata = rawdata[:]

  alldata = torch.from_numpy(rawdata).to(device)

  tensor_dataset = torch.utils.data.TensorDataset(alldata)

  train, valid, test = random_split(tensor_dataset, split, generator=torch.Generator().manual_seed(42))

  train_np = train[:][0].to('cpu').numpy()
  valid_np = valid[:][0].to('cpu').numpy()
  test_np = test[:][0].to('cpu').numpy()

  train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

  return train_loader, train_np, valid_np, test_np

def init_model(dimensionality, num_layers):

  base = nf.distributions.base.DiagGaussian(dimensionality)
  flows = []

  for i in range(num_layers):
    param_map = nf.nets.MLP([int(dimensionality/2), 64, 64, dimensionality], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(dimensionality, mode='swap'))

  model = nf.NormalizingFlow(base, flows)

  return model

def train_epoch(loader, model):

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    #for x in tqdm(loader):
    for x in loader:

      x = x[0]
      x = x.float()

      model.train()

      optimizer.zero_grad()

      loss = model.forward_kld(x)

      if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

def sampler(model, num_samples):

  model.eval()
  start = time.time()
  gendata = model.sample(num_samples)[0].detach().to('cpu').numpy()
  pass
  end = time.time()
  delta = end - start
  #print(f"Generated {num_samples} samples in {delta} seconds at a rate of {num_samples/delta} samples per second.")
  return gendata, num_samples/delta

def train_and_sample(model, loader, valid_np, test_np, pca, bounds):
  
  valid_KLD_hist = np.array([])
  valid_WD_hist = np.array([])
  
  while valid_KLD_hist.shape[0] < 5 or np.mean(valid_KLD_hist[-3:] - valid_KLD_hist[-4:-1]) < 0:
    train_epoch(loader, model)
    
    model.eval()
    
    generated_samples = sampler(model, len(valid_np))[0]
    valid_KLD = utils.counts_to_KLD(generated_samples, valid_np, pca, bounds)
    valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
    valid_WD = utils.counts_to_WD(generated_samples, valid_np, pca, bounds)
    valid_WD_hist = np.append(valid_WD_hist, valid_WD)
    #print('KLD history:', valid_KLD_hist)
    #print('WD history:', valid_WD_hist)
  
  generated_testing, speed = sampler(model, len(test_np))
  final_KLD_score = utils.counts_to_KLD(generated_testing, test_np, pca, bounds)
  final_WD_score = utils.counts_to_WD(generated_testing, test_np, pca, bounds)
  iterations = len(valid_KLD_hist)

  info = {'Training data amount': len(loader.dataset.indices), 'Learnable parameters': params, 'Iterations': iterations, 
          'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score}

  return generated_testing, info

