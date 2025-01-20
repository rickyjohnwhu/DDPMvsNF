import numpy as np
from scipy.stats import wasserstein_distance

def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Parameters:
    model (nn.Module): The model whose parameters are to be counted.

    Returns:
    int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prob_hist(data, bounds, binw=0.1):
    """
    Computes the probability histogram for 2D data.

    Parameters:
    data (np.ndarray): The 2D data array to compute the histogram for, with shape (N, 2).
    bounds (tuple): A tuple of the form (xmin, xmax, ymin, ymax) representing the bounds of the histogram.
    binw (float, optional): The bin width for the histogram. Defaults to 0.1.

    Returns:
    np.ndarray: The flattened probability histogram counts.
    """
    binner = (np.arange(bounds[0], bounds[1] + binw, binw), np.arange(bounds[2], bounds[3] + binw, binw))
    counts, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=binner, density=True)
    counts = counts.ravel()

    return counts

def KLD(a, b):
    """
    Computes the Kullback-Leibler divergence (KLD) between two distributions.

    Parameters:
    a (np.ndarray): The first distribution.
    b (np.ndarray): The second distribution.

    Returns:
    float: The computed KLD between the distributions.
    """
    a = np.asarray(a, dtype=float) + 1e-7
    b = np.asarray(b, dtype=float) + 1e-7
    return np.sum(np.where(a * b != 0, a * np.log(a / b), 0))

def counts_to_KLD(data1, data2, pca, bounds):
    """
    Computes the Kullback-Leibler divergence between two datasets after applying PCA transformation.

    Parameters:
    data1 (np.ndarray): The first dataset to be compared.
    data2 (np.ndarray): The second dataset to be compared.
    pca (PCA): The PCA model used for dimensionality reduction.
    bounds (tuple): The bounds for the histogram used to compute KLD.

    Returns:
    float: The KLD between the two datasets after PCA transformation.
    """
    pca_data1 = pca.transform(data1)
    pca_data2 = pca.transform(data2)

    counts1 = prob_hist(pca_data1, bounds)
    counts2 = prob_hist(pca_data2, bounds)

    return KLD(counts1, counts2)

def counts_to_WD(data1, data2, pca, bounds):
    """
    Computes the Wasserstein distance between two datasets after applying PCA transformation.

    Parameters:
    data1 (np.ndarray): The first dataset to be compared.
    data2 (np.ndarray): The second dataset to be compared.
    pca (PCA): The PCA model used for dimensionality reduction.
    bounds (tuple): The bounds for the histogram used to compute the Wasserstein distance.

    Returns:
    float: The Wasserstein distance between the two datasets after PCA transformation.
    """
    pca_data1 = pca.transform(data1)
    pca_data2 = pca.transform(data2)

    counts1 = prob_hist(pca_data1, bounds)
    counts2 = prob_hist(pca_data2, bounds)

    return wasserstein_distance(counts1, counts2)

def FE_hist(data, bounds, binw=0.1, binner=None):
    """
    Computes the free energy (FE) histogram for a 2D dataset and returns the free energy surface.

    Parameters:
    data (np.ndarray): The 2D dataset for which to compute the free energy.
    bounds (tuple): The bounds for the histogram, given as (xmin, xmax, ymin, ymax).
    binw (float, optional): The bin width for the histogram. Defaults to 0.1.
    binner (tuple, optional): Custom bin edges to be used for the histogram. If None, default bins are used.

    Returns:
    np.ndarray: The free energy surface as a 2D array.
    np.ndarray: The xedges of the histogram.
    np.ndarray: The yedges of the histogram.
    """
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

