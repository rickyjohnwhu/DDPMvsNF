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
    """
    A custom dataset class for handling 1D tensor data.

    Attributes:
    data (Tensor): The dataset stored as a tensor, reshaped to include a singleton channel dimension.
    """
    
    def __init__(self, datatensor):
        """
        Initializes the dataset.

        Parameters:
        datatensor (torch.Tensor): The raw tensor data to be stored in the dataset.
        """
        datatensor = datatensor.unsqueeze(1)
        self.data = datatensor

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The number of data points in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset.

        Parameters:
        idx (int): The index of the data point to retrieve.

        Returns:
        torch.Tensor: The data point at the specified index, converted to a float tensor.
        """
        x = self.data[idx]
        return x.float()


def worker_init_fn(worker_id):
    """
    Initializes each worker's random seed to ensure reproducibility.

    Parameters:
    worker_id (int): The worker's ID.
    """
    np.random.seed(args.seed + worker_id)


def preprocess_samples(filepath, split, batch_size, dataset_limiter):
    """
    Loads and preprocesses the dataset, splitting it into training, validation, and test sets.

    Parameters:
    filepath (str): The path to the dataset file (numpy format).
    split (tuple): A tuple representing the proportion for splitting the dataset (e.g., (0.8, 0.1, 0.1)).
    batch_size (int): The batch size to use for the DataLoader.
    dataset_limiter (int): The maximum number of samples to use from the dataset.

    Returns:
    DataLoader: The training data loader.
    np.ndarray: The training data (numpy array).
    np.ndarray: The validation data (numpy array).
    np.ndarray: The test data (numpy array).
    Dataset: The custom dataset containing the entire data.
    """
    rawdata = np.load(filepath)
    rawdata = rawdata[:dataset_limiter]

    alldata = torch.from_numpy(rawdata).to('cpu')

    num_set = number_dataset(alldata)

    train, valid, test = random_split(num_set, split)

    train_np = train[:].squeeze(1).to('cpu').numpy()
    valid_np = valid[:].squeeze(1).to('cpu').numpy()
    test_np = test[:].squeeze(1).to('cpu').numpy()

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)

    return train_loader, train_np, valid_np, test_np, num_set


def polynomial_noise(t, alpha_max, alpha_min, s=1e-5):
    """
    Computes the polynomial noise schedule.

    Parameters:
    t (torch.Tensor): The time steps.
    alpha_max (float): The maximum value of alpha.
    alpha_min (float): The minimum value of alpha.
    s (float): A small constant to avoid numerical issues.

    Returns:
    torch.Tensor: The computed noise schedule.
    """
    T = t[-1]
    alphas = (1 - 2 * s) * (1 - (t / T) ** 2) + s
    a = alphas[1:] / alphas[:-1]
    a[a ** 2 < 0.001] = 0.001
    alpha_schedule = torch.cumprod(a, 0)
    return alpha_schedule


NOISE_FUNCS = {
    "polynomial": polynomial_noise,
}


class VPDiffusion:
    """
    Class for implementing the diffusion process used in training and sampling.

    Attributes:
    num_diffusion_timesteps (int): The number of diffusion timesteps.
    times (torch.Tensor): The time steps used in diffusion.
    alphas (torch.Tensor): The alpha values for the noise schedule.
    bmul (function): A vectorized multiplication function.
    """

    def __init__(self, num_diffusion_timesteps, noise_schedule="polynomial", alpha_max=20., alpha_min=0.01, NOISE_FUNCS=NOISE_FUNCS):
        """
        Initializes the diffusion process.

        Parameters:
        num_diffusion_timesteps (int): Number of diffusion timesteps.
        noise_schedule (str): Type of noise schedule to use (e.g., "polynomial").
        alpha_max (float): The maximum value for alpha in the noise schedule.
        alpha_min (float): The minimum value for alpha in the noise schedule.
        NOISE_FUNCS (dict): Dictionary containing noise schedule functions.
        """
        self.bmul = vmap(torch.mul)
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.times = torch.arange(num_diffusion_timesteps)
        self.alphas = NOISE_FUNCS[noise_schedule](torch.arange(num_diffusion_timesteps + 1), alpha_max, alpha_min)

    def get_alphas(self):
        """
        Returns the computed alphas for the noise schedule.

        Returns:
        torch.Tensor: The alpha values for the noise schedule.
        """
        return self.alphas

    def forward_kernel(self, x0, t):
        """
        Computes the forward diffusion step, adding noise to the input.

        Parameters:
        x0 (torch.Tensor): The initial data at time 0.
        t (torch.Tensor): The time step.

        Returns:
        torch.Tensor: The noisy data at time t.
        torch.Tensor: The added noise.
        """
        alphas_t = self.alphas[t]
        noise = torch.randn_like(x0)
        x_t = self.bmul(x0, alphas_t.sqrt()) + self.bmul(noise, (1 - alphas_t).sqrt())
        return x_t, noise

    def reverse_kernel(self, x_t, t, backbone, pred_type):
        """
        Computes the reverse diffusion step to recover the original data.

        Parameters:
        x_t (torch.Tensor): The noisy data at time t.
        t (torch.Tensor): The current time step.
        backbone (nn.Module): The neural network model for prediction.
        pred_type (str): The type of prediction ("noise" or "x0").

        Returns:
        torch.Tensor: The predicted clean data (x0).
        torch.Tensor: The predicted noise.
        """
        alphas_t = self.alphas[t]

        if pred_type == "noise":
            noise = backbone(x_t, alphas_t)
            noise_interp = self.bmul(noise, (1 - alphas_t).sqrt())
            x0_t = self.bmul((x_t - noise_interp), 1 / alphas_t.sqrt())
            
        elif pred_type == "x0":
            x0_t = backbone(x_t, alphas_t)
            x0_interp = self.bmul(x0_t, alphas_t.sqrt())
            noise = self.bmul((x_t - x0_interp), 1 / (1 - alphas_t).sqrt())
        
        else:
            raise Exception("Please provide a valid prediction type: 'noise' or 'x0'")

        return x0_t, noise

    def reverse_step(self, x_t, t, t_next, backbone, pred_type):
        """
        Performs a single step of reverse diffusion.

        Parameters:
        x_t (torch.Tensor): The noisy data at time t.
        t (torch.Tensor): The current time step.
        t_next (torch.Tensor): The next time step.
        backbone (nn.Module): The model to use for prediction.
        pred_type (str): The type of prediction ("noise" or "x0").

        Returns:
        torch.Tensor: The next step of the reversed data.
        """
        alphas_t = self.alphas[t]
        alphas_t_next = self.alphas[t_next]
        sigmas_t = ((1 - alphas_t_next) / (1 - alphas_t)).sqrt() * (1 - alphas_t / alphas_t_next).sqrt()

        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type)

        output_shape = x0_t.size()
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul((1 - alphas_t_next - (sigmas_t ** 2)).sqrt(), noise) + self.bmul(sigmas_t, torch.randn(output_shape))
        return xt_next

    def sample_prior(self, xt):
        """
        Samples a random noise tensor to start the generation process.

        Parameters:
        xt (torch.Tensor): The input tensor (usually noisy data).

        Returns:
        torch.Tensor: The generated noise tensor.
        """
        noise = torch.randn_like(xt)
        return noise


def init_model_backbone(loader, resnet_block_groups, learned_sinusoidal_cond, learned_sinusoidal_dim, dim_mults, num_set, lr, model_dim):
    """
    Initializes the model and its backbone.

    Parameters:
    loader (DataLoader): The training data loader.
    resnet_block_groups (int): Number of groups for resnet blocks in the U-Net.
    learned_sinusoidal_cond (bool): Whether to use learned sinusoidal conditioning.
    learned_sinusoidal_dim (int): The dimensionality of the sinusoidal conditioning.
    dim_mults (list): The list of dimension multipliers for U-Net layers.
    num_set (Dataset): The dataset object.
    lr (float): The learning rate.
    model_dim (int): The dimensionality of the model.

    Returns:
    nn.Module: The U-Net model.
    nn.Module: The backbone model for diffusion.
    int: The number of torsions (features) in the dataset.
    """
    num_torsions = loader.dataset[0].shape[-1]
    
    model = Unet1D(dim=model_dim, channels=1, resnet_block_groups=resnet_block_groups,
                   learned_sinusoidal_cond=learned_sinusoidal_cond,
                   learned_sinusoidal_dim=learned_sinusoidal_dim, dim_mults=dim_mults)

    backbone = ConvBackbone1D(model=model, data_shape=num_torsions, target_shape=model_dim, num_dims=len(num_set.data.shape), lr=lr)
    
    return model, backbone, num_torsions


def l2_loss(x, x_pred):
    """
    Computes the L2 loss between the true and predicted values.

    Parameters:
    x (torch.Tensor): The true data.
    x_pred (torch.Tensor): The predicted data.

    Returns:
    torch.Tensor: The computed L2 loss.
    """
    return (x - x_pred).pow(2).sum((1, 2)).pow(0.5).mean()


def train_epoch(loader, backbone, diffusion):
    """
    Trains the model for one epoch.

    Parameters:
    loader (DataLoader): The data loader for training data.
    backbone (nn.Module): The model backbone to update during training.
    diffusion (VPDiffusion): The diffusion process to use during training.
    """
    for b in loader:
        t = torch.randint(low=0, high=diffusion.num_diffusion_timesteps, size=(b.size(0),)).long()
      
        b_t, e_0 = diffusion.forward_kernel(b, t)
        b_0, e_t = diffusion.reverse_kernel(b_t, t, backbone, "x0")

        loss = l2_loss(b, b_0)
        backbone.optim.zero_grad()
        loss.backward()
        backbone.optim.step()


def sample_batch(batch_size, loader, diffusion, backbone, num_set, pred_type="x0"):
    """
    Samples a batch of generated data using the diffusion process.

    Parameters:
    batch_size (int): The batch size for sampling.
    loader (DataLoader): The data loader for training data.
    diffusion (VPDiffusion): The diffusion process to use during sampling.
    backbone (nn.Module): The model backbone used for prediction.
    num_set (Dataset): The dataset object.
    pred_type (str): The prediction type ("x0" or "noise").

    Returns:
    torch.Tensor: The generated batch of data.
    """
    def sample_prior(batch_size, shape):
        """
        Samples random noise as the prior.

        Parameters:
        batch_size (int): The number of samples to generate.
        shape (torch.Size): The shape of the generated samples.

        Returns:
        torch.Tensor: The generated random noise.
        """
        prior_sample = torch.randn(batch_size, *shape[1:], dtype=torch.float)
        return prior_sample

    def get_adjacent_times(times):
        """
        Returns pairs of adjacent time steps for reverse diffusion.

        Parameters:
        times (torch.Tensor): The list of time steps.

        Returns:
        list: A list of tuples containing adjacent time steps.
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


def sample_loop(num_samples, batch_size, loader, diffusion, backbone, num_torsions, num_set):
    """
    Generates a loop of samples from the trained model.

    Parameters:
    num_samples (int): The number of samples to generate.
    batch_size (int): The batch size for sampling.
    loader (DataLoader): The data loader for training data.
    diffusion (VPDiffusion): The diffusion process to use.
    backbone (nn.Module): The model backbone for prediction.
    num_torsions (int): The number of features.
    num_set (Dataset): The dataset object.

    Returns:
    torch.Tensor: The generated samples.
    """
    gendata = torch.empty(0, 1, num_torsions)

    n_runs = max(num_samples // batch_size, 1)
    if num_samples <= batch_size:
        batch_size = num_samples

    with torch.no_grad():
        for save_idx in range(n_runs):
            x0 = sample_batch(batch_size, loader, diffusion, backbone, num_set)
            gendata = torch.cat((gendata, x0), 0)

    return gendata


def sampler(samples, batchsize, loader, diffusion, backbone, num_torsions, num_set):
    """
    Generates a set of samples using the trained model and diffusion process.

    Parameters:
    samples (int): The number of samples to generate.
    batchsize (int): The batch size for sampling.
    loader (DataLoader): The data loader for training data.
    diffusion (VPDiffusion): The diffusion process to use.
    backbone (nn.Module): The model backbone for prediction.
    num_torsions (int): The number of features.
    num_set (Dataset): The dataset object.

    Returns:
    torch.Tensor: The generated samples.
    float: The speed of sample generation in samples per second.
    """
    start = time.time()
    gendata = sample_loop(samples, batchsize, loader, diffusion, backbone, num_torsions, num_set)
    end = time.time()
    delta = end - start
    gendata = gendata.squeeze(1)
    return gendata, samples / delta


def train_and_sample(model, loader, valid_np, test_np, train_np, pca, bounds, diffusion, backbone, num_torsions, num_set, sample_batching, epoch_max, aib9_status):
    """
    Trains the model and generates samples.

    Parameters:
    model (nn.Module): The model to train.
    loader (DataLoader): The training data loader.
    valid_np (np.ndarray): The validation data.
    test_np (np.ndarray): The test data.
    train_np (np.ndarray): The training data.
    pca (PCA): The PCA object for dimensionality reduction.
    bounds (tuple): Bounds for KLD and WD computation.
    diffusion (VPDiffusion): The diffusion process used during training and sampling.
    backbone (nn.Module): The model backbone.
    num_torsions (int): The number of features in the data.
    num_set (Dataset): The dataset object.
    sample_batching (int): The number of samples per batch during sampling.
    epoch_max (int): The maximum number of epochs to train.
    aib9_status (bool): Whether to calculate residue KLD.

    Returns:
    torch.Tensor: The generated samples.
    dict: A dictionary containing performance metrics such as KLD, WD, and time information.
    """
    fullstart = time.time()

    epochs = 0
    while epochs < epoch_max:
        train_epoch(loader, backbone, diffusion)
        epochs += 1

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
    iterations = epochs

    fullstop = time.time()

    if aib9_status:
        info = {
            'Architecture': 'DDPM',
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
            'Architecture': 'DDPM',
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
