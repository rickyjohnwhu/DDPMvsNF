import torch
import numpy as np
import normflows as nf
import time
import matplotlib.pyplot as plt
import pprint
import utils
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from torch import nn
import random

def get_info(model, dimension, modes, keyword):
    info = np.load(f"data_output/{model}_data/info_dim_{dimension}_modes_{modes}_{keyword}.npz", allow_pickle=True)
    info = info['arr_0']
    infodict = dict(np.reshape(info, (1,))[0])
    return infodict

def get_old_info(model, dimension, modes, keyword):
    info = np.load(f"data_output_old/{model}_data/info_dim_{dimension}_modes_{modes}_{keyword}.npz", allow_pickle=True)
    info = info['arr_0']
    infodict = dict(np.reshape(info, (1,))[0])
    return infodict

def get_values_from_dim(model, dim_array, modes, keyword, dict_key):
    values_array = np.array([])
    for dim in dim_array:
        values_array = np.append(values_array, get_info(model, dim, modes, keyword)[dict_key])
    return values_array

def get_values_from_modes(model, dim, modes_array, keyword, dict_key):
    values_array = np.array([])
    for mode in modes_array:
        values_array = np.append(values_array, get_info(model, dim, mode, keyword)[dict_key])
    return values_array

def get_values_from_keyword(model, dim, modes, keyword_array, dict_key):
    values_array = np.array([])
    for keyword in keyword_array:
        values_array = np.append(values_array, get_info(model, dim, modes, keyword)[dict_key])
    return values_array

def construct_iterated_keyword_array(base_array, base_keyword, iteration):
    keyword_array = np.array([])
    for base in base_array:
        keyword_array = np.append(keyword_array, f"{base}_{base_keyword}_{iteration}")
    return keyword_array

def get_training_data(dimension, modes):
    data = np.load(f"data_input/dim_{dimension}_modes_{modes}.npy", allow_pickle=True)
    return data

def get_data(model, dimension, modes, keyword):
    data = np.load(f"data_output/{model}_data/data_dim_{dimension}_modes_{modes}_{keyword}.npy", allow_pickle=True)
    return data

def get_aib9_info(model, residue, data_type, keyword):
    info = np.load(f"data_output/{model}_data/info_aib9_{data_type}_{residue}_{keyword}.npz", allow_pickle=True)
    info = info['arr_0']
    infodict = dict(np.reshape(info, (1,))[0])
    return infodict

def get_aib9_values_from_res(model, residue_array, data_type, keyword, dict_key):
    values_array = np.array([])
    for residue in residue_array:
        values_array = np.append(values_array, get_aib9_info(model, residue, data_type, keyword)[dict_key])
    return values_array

def get_aib9_values_from_keyword(model, residue, data_type, keyword_array, dict_key):
    values_array = np.array([])
    for keyword in keyword_array:
        values_array = np.append(values_array, get_aib9_info(model, residue, data_type, keyword)[dict_key])
    return values_array

def get_aib9_data(model, residue, data_type, keyword):
    data = np.load(f"data_output/{model}_data/data_aib9_{data_type}_{residue}_{keyword}.npy", allow_pickle=True)
    return data

def get_aib9_training_data(residue, data_type):
    data = np.load(f"data_input/aib9_{data_type}_{residue}.npy", allow_pickle=True)
    return data

def old_baseline_maker_aib9(residue, dataset_limiter, split, assumed_modes, bounds):
    total_data = np.load(f"data_input/aib9_total_0.npy", allow_pickle=True)[:, int(residue*2):int((residue+1)*2)]
    total_shuffle = shuffler(total_data[:dataset_limiter])
    total_len = total_shuffle.shape[0]
    train_data = total_shuffle[:int(split[0]*total_len)]
    gmm = GaussianMixture(assumed_modes)
    fitted = gmm.fit(train_data)
    test_data = total_shuffle[:int(split[1]*total_len)]
    generated_data = fitted.sample(int(split[1]*total_len))[0]
    pca = PCA(n_components = 2)
    pca.fit_transform(train_data)
    final_score = utils.counts_to_KLD(generated_data, test_data, pca, bounds)
    return final_score, generated_data

def baseline_maker_aib9(residue, dataset_limiter, split, assumed_modes, bounds):
    total_data = np.load(f"data_input/aib9_total_0.npy", allow_pickle=True)
    total_shuffle = shuffler(total_data[:dataset_limiter])
    total_len = total_shuffle.shape[0]
    train_data = total_shuffle[:int(split[0]*total_len)]
    gmm = GaussianMixture(assumed_modes)
    fitted = gmm.fit(train_data)
    test_data = total_shuffle[:int(split[1]*total_len)]
    generated_data = fitted.sample(int(split[1]*total_len))[0][:, int(residue*2):int((residue+1)*2)]
    pca = PCA(n_components = 2)
    pca.fit_transform(train_data[:, int(residue*2):int((residue+1)*2)])
    final_score = utils.counts_to_KLD(generated_data, test_data[:, int(residue*2):int((residue+1)*2)], pca, bounds)
    return final_score, generated_data

def find_nan_inf(array):
    nan_indices = np.isnan(array)
    inf_indices = np.isinf(array)
    nan_values = array[nan_indices]
    inf_values = array[inf_indices]
    return nan_values, inf_values

def shuffler(array):
    mask = np.arange(len(array))
    random.shuffle(mask)
    return array[mask]