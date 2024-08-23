import numpy as np
from scipy.stats import wasserstein_distance

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
