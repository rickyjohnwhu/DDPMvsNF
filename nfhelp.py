import torch
import numpy as np
import normflows as nf
import time
import matplotlib.pyplot as plt
import pprint
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prob_hist(data, bounds, binw=0.1):

    binner = (np.arange(bounds[0], bounds[1] + binw, binw), np.arange(bounds[0], bounds[1] + binw, binw))

    counts, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=binner, density=True)
    counts = counts.ravel()

    return counts

def KLD(a, b):

    a = np.asarray(a, dtype=float) + 1e-7
    b = np.asarray(b, dtype=float) + 1e-7

    return np.sum(np.where(a*b != 0, a * np.log(a / b), 0))

def counts_to_KLD(data1, data2, pca, bounds):

    pca_data1 = pca.transform(data1)
    pca_data2 = pca.transform(data2)

    counts1 = prob_hist(pca_data1, bounds)
    counts2 = prob_hist(pca_data2, bounds)

    return KLD(counts1, counts2)

def counts_to_WD(data1, data2, pca, bounds):

    pca_data1 = pca.transform(data1)
    pca_data2 = pca.transform(data2)

    counts1 = prob_hist(pca_data1, bounds)
    counts2 = prob_hist(pca_data2, bounds)

    return wasserstein_distance(counts1, counts2)

def FE_hist(data, bounds, binw=0.1, binner=None):

    if binner is None:
        binner = (np.arange(bounds[0], bounds[1] + binw, binw),
                    np.arange(bounds[2], bounds[3] + binw, binw))

    counts, xedges, yedges = np.histogram2d(data[:,0], data[:,1],
                                           bins=binner, density=True)
    prob = counts + 1e-3
    G = -np.log(prob)
    G[G == np.inf] = -1
    G[G == -1] = max(G.ravel())
    G -= min(G.ravel())
    
    return G, xedges, yedges

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
    valid_KLD = counts_to_KLD(generated_samples, valid_np, pca, bounds)
    valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
    valid_WD = counts_to_WD(generated_samples, valid_np, pca, bounds)
    valid_WD_hist = np.append(valid_WD_hist, valid_WD)
    #print('KLD history:', valid_KLD_hist)
    #print('WD history:', valid_WD_hist)
  
  generated_testing, speed = sampler(model, len(test_np))
  final_KLD_score = counts_to_KLD(generated_testing, test_np, pca, bounds)
  final_WD_score = counts_to_WD(generated_testing, test_np, pca, bounds)
  iterations = len(valid_KLD_hist)

  info = {'Training data amount': len(loader.dataset.indices), 'Dimensions': dimensions, 'Learnable parameters': params, 'Iterations': iterations, 
          'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score}

  return generated_testing, info

