import torch
import numpy as np
import normflows as nf
import time
import matplotlib.pyplot as plt
import pprint
import utils
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from sklearn.decomposition import PCA

def preprocess_samples(filepath, split, batch_size, dataset_limiter, device):
    """
    Preprocesses the dataset by loading the raw data, splitting it into train, validation, and test sets,
    and creating a data loader for training.

    Parameters:
    filepath (str): Path to the dataset file (assumed to be a numpy array).
    split (tuple): A tuple with the proportions of train, validation, and test sets.
    batch_size (int): The batch size to use for training.
    dataset_limiter (int): The number of samples to use from the dataset (for limiting the dataset size).
    device (torch.device): The device to load the data onto (CPU or GPU).

    Returns:
    DataLoader: The training data loader.
    np.ndarray: The training data as a numpy array.
    np.ndarray: The validation data as a numpy array.
    np.ndarray: The test data as a numpy array.
    """
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


def init_model(dimensionality, num_layers, num_blocks, channels, context, device):
    """
    Initializes the normalizing flow model with the given configuration.

    Parameters:
    dimensionality (int): The dimensionality of the data.
    num_layers (int): The number of flow layers.
    num_blocks (int): The number of blocks per flow layer.
    channels (int): The number of channels for each block.
    context (bool): Whether to use context for the normalizing flow.
    device (torch.device): The device to load the model onto (CPU or GPU).

    Returns:
    nf.NormalizingFlow: The initialized normalizing flow model.
    """
    base = nf.distributions.base.DiagGaussian(dimensionality)

    flow_layers = []
    for i in range(num_layers):
        flow_layers += [nf.flows.AutoregressiveRationalQuadraticSpline(dimensionality, num_blocks, channels, permute_mask=True)]

    model = nf.NormalizingFlow(base, flow_layers)
    model.to(device)

    return model


def train_epoch(loader, model, lr):
    """
    Performs one epoch of training on the model.

    Parameters:
    loader (DataLoader): The data loader for the training data.
    model (nf.NormalizingFlow): The normalizing flow model to train.
    lr (float): The learning rate for the optimizer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()

    for x in loader:
        x = x[0].float()

        optimizer.zero_grad()

        loss = model.forward_kld(x)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()


def sampler(model, num_samples, small_sample, dim):
    """
    Samples data from the trained normalizing flow model.

    Parameters:
    model (nf.NormalizingFlow): The trained normalizing flow model.
    num_samples (int): The number of samples to generate.
    small_sample (int): The batch size for each sampling step.
    dim (int): The dimensionality of the data.

    Returns:
    np.ndarray: The generated samples.
    float: The speed of sample generation in samples per second.
    """
    model.eval()
    start = time.time()
    gendata = np.zeros((num_samples, dim))

    for i in range(int(num_samples / small_sample)):
        genstep = model.sample(small_sample)[0].detach().to('cpu').numpy()
        gendata[i * small_sample:(i + 1) * small_sample] = genstep

    end = time.time()
    delta = end - start
    return gendata, num_samples / delta


def train_and_sample(model, loader, valid_np, test_np, train_np, pca, bounds, lr, epoch_max, small_sample, dim, aib9_status):
    """
    Trains the model and generates samples after training.

    Parameters:
    model (nf.NormalizingFlow): The normalizing flow model to train.
    loader (DataLoader): The data loader for the training data.
    valid_np (np.ndarray): The validation data (for evaluation purposes).
    test_np (np.ndarray): The test data (for evaluation purposes).
    train_np (np.ndarray): The training data (for evaluation purposes).
    pca (PCA): PCA object for dimensionality reduction (for evaluation).
    bounds (tuple): Bounds for calculating KLD and WD scores.
    lr (float): The learning rate for training.
    epoch_max (int): The maximum number of epochs to train for.
    small_sample (int): The number of samples to generate at a time.
    dim (int): The dimensionality of the data.
    aib9_status (bool): Whether to calculate residue KLD scores.

    Returns:
    np.ndarray: The generated samples.
    dict: A dictionary containing performance metrics like KLD, WD, and training time.
    """
    fullstart = time.time()

    epochs = 0
    while epochs < epoch_max:
        train_epoch(loader, model, lr)
        epochs += 1

    model.eval()
    generated_testing, speed = sampler(model, len(test_np), small_sample, dim)

    if aib9_status:
        res_array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        res_KLD_array = np.array([])

        for res in res_array:
            KLD_pca = PCA(n_components=2)
            training_KLD_pca = KLD_pca.fit_transform(train_np[:, int(res * 2):int((res + 1) * 2)])
            res_KLD_score = utils.counts_to_KLD(generated_testing[:, int(res * 2):int((res + 1) * 2)], test_np[:, int(res * 2):int((res + 1) * 2)], KLD_pca, bounds)
            res_KLD_array = np.append(res_KLD_array, res_KLD_score)

    final_KLD_score = utils.counts_to_KLD(generated_testing, test_np, pca, bounds)
    final_WD_score = utils.counts_to_WD(generated_testing, test_np, pca, bounds)
    iterations = epochs

    fullstop = time.time()

    if aib9_status:
        info = {
            'Architecture': 'NS',
            'Training data amount': len(loader.dataset.indices),
            'Learnable parameters': utils.count_parameters(model),
            'Iterations': iterations,
            'Speed (samples/s)': speed,
            'Final KLD': final_KLD_score,
            'Final WD': final_WD_score,
            'Dimensions': len(generated_testing[0]),
            'Total train/sample time': fullstop - fullstart,
            'Residue KLD': res_KLD_array
        }
    else:
        info = {
            'Architecture': 'NS',
            'Training data amount': len(loader.dataset.indices),
            'Learnable parameters': utils.count_parameters(model),
            'Iterations': iterations,
            'Speed (samples/s)': speed,
            'Final KLD': final_KLD_score,
            'Final WD': final_WD_score,
            'Dimensions': len(generated_testing[0]),
            'Total train/sample time': fullstop - fullstart
        }

    return generated_testing, info
