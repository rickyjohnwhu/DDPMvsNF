import torch
import numpy as np
import normflows as nf
import time
import matplotlib.pyplot as plt
import pprint
import utils
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def preprocess_samples(filepath, split, batch_size, dataset_limiter, device):

    rawdata = np.load(filepath)
    rawdata = rawdata[:dataset_limiter]
    
    alldata = torch.from_numpy(rawdata).to(device)

    #alldata = torch.from_numpy(rawdata)

    tensor_dataset = torch.utils.data.TensorDataset(alldata)

    train, valid, test = random_split(tensor_dataset, split)

    train_np = train[:][0].to('cpu').numpy()
    valid_np = valid[:][0].to('cpu').numpy()
    test_np = test[:][0].to('cpu').numpy()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    return train_loader, train_np, valid_np, test_np

def init_model(dimensionality, num_layers, num_blocks, channels, context, device):

    base = nf.distributions.base.DiagGaussian(dimensionality)
    
    flow_layers = []
    for i in range(num_layers):
        flow_layers += [nf.flows.AutoregressiveRationalQuadraticSpline(dimensionality, num_blocks, channels, context, permute_mask=True)]

    model = nf.NormalizingFlow(base, flow_layers)
    
    model.to(device)

    return model

def train_epoch(loader, model, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()

    for x in loader:

        x = x[0]
        x = x.float()

        optimizer.zero_grad()

        loss = model.forward_kld(x)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

def sampler(model, num_samples, small_sample, dim):

    model.eval()
    start = time.time()
    #gendata = model.sample(num_samples)[0].detach().to('cpu').numpy()
    gendata = np.zeros((num_samples, dim))
    for i in range(int(num_samples/small_sample)):
        genstep = model.sample(small_sample)[0].detach().to('cpu').numpy()
        gendata[i*small_sample:(i+1)*small_sample] = genstep
        #tensor = torch.cuda.FloatTensor(1)
        #allocated_memory = torch.cuda.memory_allocated()
        #print(f"Currently allocated CUDA memory: {allocated_memory / 1024**3:.2f} GB")
    pass
    end = time.time()
    delta = end - start
    #print(f"Generated {num_samples} samples in {delta} seconds at a rate of {num_samples/delta} samples per second.")
    return gendata, num_samples/delta

def train_and_sample(model, loader, valid_np, test_np, train_np, pca, bounds, lr, epoch_max, small_sample, dim, aib9_status):

    #valid_KLD_hist = np.array([])
    #valid_WD_hist = np.array([])

    fullstart = time.time()
    
    #while valid_KLD_hist.shape[0] < 5 or np.mean(valid_KLD_hist[-3:] - valid_KLD_hist[-4:-1]) < 0:
    epochs = 0
    while epochs < epoch_max:
      
        train_epoch(loader, model, lr)

        epochs += 1

        #model.eval()

        #generated_samples = sampler(model, len(valid_np))[0]
        #valid_KLD = utils.counts_to_KLD(generated_samples, valid_np, pca, bounds)
        #valid_KLD_hist = np.append(valid_KLD_hist, valid_KLD)
        #valid_WD = utils.counts_to_WD(generated_samples, valid_np, pca, bounds)
        #valid_WD_hist = np.append(valid_WD_hist, valid_WD)
        #print('KLD history:', valid_KLD_hist)
        #print('WD history:', valid_WD_hist)

    model.eval()
    generated_testing, speed = sampler(model, len(test_np), small_sample, dim)
    if aib9_status:
        res_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        res_KLD_array = np.array([])
        for res in res_array:
            KLD_pca = PCA(n_components=2)
            training_KLD_pca = KLD_pca.fit_transform(train_np[:, int(res*2):int((res+1)*2)])
            res_KLD_score = utils.counts_to_KLD(generated_testing[:, int(res*2):int((res+1)*2)], test_np[:, int(res*2):int((res+1)*2)], KLD_pca, bounds)
            res_KLD_array = np.append(res_KLD_array, res_KLD_score)
    
    #print('NaN found:', generated_testing.flatten()[np.isnan(generated_testing.flatten())].shape[0])
    
    final_KLD_score = utils.counts_to_KLD(generated_testing, test_np, pca, bounds)
    final_WD_score = utils.counts_to_WD(generated_testing, test_np, pca, bounds)
    #iterations = len(valid_KLD_hist)
    iterations = epochs

    fullstop = time.time()

    if aib9_status:
        info = {'Architecture': 'NS', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart, 'Residue KLD': res_KLD_array}
    else:
        info = {'Architecture': 'NS', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart}

    return generated_testing, info