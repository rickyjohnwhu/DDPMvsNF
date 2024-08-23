import numpy as np
import os
import argparse
from sklearn.decomposition import PCA
import utils_ns

def run(args):
    
    train_loader, train_np, valid_np, test_np = utils_ns.preprocess_samples(args.input, args.split, args.batch_size, args.dataset_limiter, args.device)
    model = utils_ns.init_model(args.path_dim, args.layers, args.blocks, args.channels, args.context, args.device)
    pca = PCA(n_components=2)
    trainingpca = pca.fit_transform(train_np)
    generated_testing, info = utils_ns.train_and_sample(model, train_loader, valid_np, test_np, train_np, pca, args.bounds, args.lr, args.epoch_max, args.small_sample, args.path_dim, args.aib9_status)
    args_dict = {key: getattr(args, key) for key in vars(args)}
    info_dict = {**info, **args_dict}
    np.save(args.data_output, generated_testing)
    np.savez(args.info_output, info_dict)

def main():
    
    parser = argparse.ArgumentParser(description='Description of your script.')
    parser.add_argument('--dataset_limiter', type=int, default=100000, help='Description for dataset_limiter')
    parser.add_argument('--path_dim', type=int, default=10, help='Description for path_dim')
    parser.add_argument('--path_modes', type=int, default=4, help='Description for path_modes')
    parser.add_argument('--split', type=float, nargs='+', default=[0.8, 0.1, 0.1], help='Description for split')
    parser.add_argument('--batch_size', type=int, default=512, help='Description for batch_size')
    parser.add_argument('--bounds', type=float, nargs='+', default=[-12, 12, -12, 12], help='Description for bounds')
    parser.add_argument('--lr', type=float, default=5e-4, help='Description for lr')
    parser.add_argument('--layers', type=int, default=16, help='Description for layers')
    parser.add_argument('--blocks', type=int, default=1, help='Description for blocks')
    parser.add_argument('--channels', type=int, default=32, help='Description for channels')
    parser.add_argument('--context', type=bool, default=None, help='Description for context')
    parser.add_argument('--device', type=str, default='cuda:0', help='Description for device')
    parser.add_argument('--epoch_max', type=int, default=30, help='Description for epoch_max')
    parser.add_argument('--small_sample', type=int, default=100, help='Description for small_sample')
    parser.add_argument('--aib9_status', type=bool, default=False, help='Description for aib9_status')
    parser.add_argument('--data_type', type=str, default='raw', help='Description for data_type')
    parser.add_argument('--residue', type=int, default=1, help='Description for residue')
    parser.add_argument('--keyword', type=str, default='test', help='Description for keyword')
    parser.add_argument('--input', type=str, default='error: check argparse', help='Description for input')
    parser.add_argument('--data_output', type=str, default='error: check argparse', help='Description for data_output')
    parser.add_argument('--info_output', type=str, default='error: check argparse', help='Description for info_output')
    
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()