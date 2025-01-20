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
    """
    A utility class for interpolating data between two shapes (data shape and target shape).
    This is used to upsample and downsample tensors during training and sampling.

    Attributes:
    data_shape (tuple): Original shape of the data.
    target_shape (tuple): The target shape to interpolate to.
    """
    
    def __init__(self, data_shape: tuple, target_shape: tuple):
        """
        Initializes the interpolater with data shape and target shape.
        
        Parameters:
        data_shape (tuple): Original shape of the data.
        target_shape (tuple): The target shape to interpolate to.
        """
        self.data_shape, self.target_shape = data_shape, target_shape

    def to_target(self, x):
        """
        Upsample the input tensor to the target shape using nearest neighbor interpolation.

        Parameters:
        x (torch.Tensor): The input tensor to interpolate.

        Returns:
        torch.Tensor: Interpolated tensor with the target shape.
        """
        return F.interpolate(x, size=self.target_shape, mode='nearest-exact')

    def from_target(self, x):
        """
        Downsample the input tensor to the original data shape using nearest neighbor interpolation.

        Parameters:
        x (torch.Tensor): The input tensor to interpolate.

        Returns:
        torch.Tensor: Interpolated tensor with the original shape.
        """
        return F.interpolate(x, size=self.data_shape, mode='nearest-exact')


def preprocess_samples(filepath, split, batch_size, dataset_limiter, device):
    """
    Preprocess the raw data by loading, splitting, and creating DataLoader objects.
    
    Parameters:
    filepath (str): Path to the raw data file (numpy format).
    split (tuple): Proportions for splitting the dataset into training, validation, and test sets.
    batch_size (int): The batch size to use in training.
    dataset_limiter (int): The number of samples to use from the dataset.
    device (str): The device to move the tensors to (e.g., 'cpu', 'cuda').

    Returns:
    DataLoader: DataLoader for training data.
    np.ndarray: NumPy array of training data.
    np.ndarray: NumPy array of validation data.
    np.ndarray: NumPy array of test data.
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


def init_model(dimensionality, w, sigma, model_style, time_varying, resnet_block_groups, learned_sinusoidal_cond, learned_sinusoidal_dim, dim_mults, model_dim, device):
    """
    Initializes the model, flow matcher, and base distribution.
    
    Parameters:
    dimensionality (int): Dimensionality of the data.
    w (float): Hyperparameter for model initialization.
    sigma (float): Hyperparameter for model initialization.
    model_style (str): Type of model to use ('ConditionalFlowMatcher', etc.).
    time_varying (bool): Whether the model is time-varying.
    resnet_block_groups (int): Number of groups in ResNet blocks.
    learned_sinusoidal_cond (bool): Whether to use learned sinusoidal conditioning.
    learned_sinusoidal_dim (int): Dimension of learned sinusoidal conditioning.
    dim_mults (list): Dimensionality multipliers for the model.
    model_dim (int): Dimensionality of the model.
    device (str): The device to initialize the model on (e.g., 'cpu', 'cuda').

    Returns:
    torch.nn.Module: The initialized model.
    object: The flow matcher object.
    object: The base distribution object.
    """
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
    """
    Train the model for one epoch.

    Parameters:
    loader (DataLoader): DataLoader for the training data.
    model (torch.nn.Module): The neural network model.
    FM (object): Flow matcher object.
    base (object): Base distribution object.
    lr (float): Learning rate.
    interp (Interpolater): Interpolation object to adjust sample sizes.

    Returns:
    None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_hist = np.ndarray([])

    for x in loader:
        x = x[0].float()
        
        optimizer.zero_grad()
        x0 = base.sample(len(x))
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x)

        xt = xt.unsqueeze(1)
        xt_up = interp.to_target(xt)
        vt_up = model(xt_up, t)
        vt = interp.from_target(vt_up)
        vt = vt.squeeze(1)

        loss = torch.mean((vt - ut) ** 2)

        loss.backward()
        optimizer.step()
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())


def sample(model, base, num_samples, interp):
    """
    Generate samples using the trained model.

    Parameters:
    model (torch.nn.Module): The trained model.
    base (object): Base distribution for sampling.
    num_samples (int): The number of samples to generate.
    interp (Interpolater): Interpolation object to adjust sample sizes.

    Returns:
    torch.Tensor: The generated samples.
    """
    t_span = torch.linspace(0, 1, 100)
    dt = (t_span[-1] - t_span[0]) / (len(t_span) - 1)

    xt = base.sample(num_samples)

    with torch.no_grad():
        for t in t_span:
            xt = xt.unsqueeze(1)
            xt_up = interp.to_target(xt)
        
            vt_up = model(xt_up, torch.full((xt.shape[0],), t).to('cuda:0'))
            vt = interp.from_target(vt_up)
            
            xt = xt.squeeze(1)
            vt = vt.squeeze(1)

            xt += vt * dt

    return xt


def sampler(model, base, samples, interp):
    """
    Sample new data points from the model and measure time performance.

    Parameters:
    model (torch.nn.Module): The trained model.
    base (object): Base distribution for sampling.
    samples (int): The number of samples to generate.
    interp (Interpolater): Interpolation object.

    Returns:
    np.ndarray: The generated samples.
    float: Speed in samples per second.
    """
    model.eval()
    start = time.time()

    gendata = sample(model, base, samples, interp).detach().to('cpu').numpy()

    end = time.time()
    delta = end - start
    
    return gendata, samples / delta


def train_and_sample(model, FM, base, interp, loader, valid_np, test_np, train_np, pca, bounds, lr, epoch_max, aib9_status):
    """
    Train the model and generate samples, then evaluate the performance using KLD and WD scores.

    Parameters:
    model (torch.nn.Module): The trained model.
    FM (object): Flow matcher object.
    base (object): Base distribution object.
    interp (Interpolater): Interpolation object.
    loader (DataLoader): DataLoader for the training data.
    valid_np (np.ndarray): Validation data (not used in this function directly).
    test_np (np.ndarray): Test data for evaluation.
    train_np (np.ndarray): Training data (not used directly here).
    pca (PCA): PCA object for dimensionality reduction.
    bounds (tuple): Bounds for KLD and WD calculation.
    lr (float): Learning rate.
    epoch_max (int): Maximum number of epochs for training.
    aib9_status (bool): Flag to indicate if residue KLD calculation is needed.

    Returns:
    np.ndarray: Generated testing samples.
    dict: Performance metrics including KLD, WD, and other details.
    """
    fullstart = time.time()

    epochs = 0
    while epochs < epoch_max:
        train_epoch(loader, model, FM, base, lr, interp)
        epochs += 1

    model.eval()
    generated_testing, speed = sampler(model, base, len(test_np), interp)
    
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
    iterations = epochs

    fullstop = time.time()

    if aib9_status:
        info = {'Architecture': 'CFM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart, 'Residue KLD': res_KLD_array}
    else:
        info = {'Architecture': 'CFM', 'Training data amount': len(loader.dataset.indices), 'Learnable parameters': utils.count_parameters(model), 'Iterations': iterations, 'Speed (samples/s)': speed, 'Final KLD': final_KLD_score, 'Final WD': final_WD_score, 'Dimensions': len(generated_testing[0]), 'Total train/sample time': fullstop - fullstart}

    return generated_testing, info
