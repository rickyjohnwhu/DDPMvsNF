import os
import numpy as np
import argparse
from sklearn.decomposition import PCA
import utils_cfm

def run(args):

    train_loader, train_np, valid_np, test_np = utils_cfm.preprocess_samples(args.input, args.split, args.batch_size, args.dataset_limiter, args.device)
    model, FM, base = utils_cfm.init_model(args.path_dim, args.w, args.sigma, args.model_style, args.time_varying, args.resnet_block_groups, args.learned_sinusoidal_cond, args.learned_sinusoidal_dim, args.dim_mults, args.model_dim, args.device)
    interp = utils_cfm.Interpolater((args.path_dim,), (args.model_dim,))
    pca = PCA(n_components=2)
    trainingpca = pca.fit_transform(train_np)
    generated_testing, info = utils_cfm.train_and_sample(model, FM, base, interp, train_loader, valid_np, test_np, train_np, pca, args.bounds, args.lr, args.epoch_max, args.aib9_status)
    args_dict = {key: getattr(args, key) for key in vars(args)}
    info_dict = {**info, **args_dict}
    np.save(args.data_output, generated_testing)
    np.savez(args.info_output, info_dict)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_limiter', type=int, default=100000, help='Description for dataset_limiter')
    parser.add_argument('--path_dim', type=int, default=10, help='Description for path_dim')
    parser.add_argument('--path_modes', type=int, default=6, help='Description for path_modes')
    parser.add_argument('--split', type=float, nargs='+', default=[0.8, 0.1, 0.1], help='Description for split')
    parser.add_argument('--batch_size', type=int, default=512, help='Description for batch_size')
    parser.add_argument('--bounds', type=float, nargs='+', default=[-12, 12, -12, 12], help='Description for bounds')
    parser.add_argument('--lr', type=float, default=5e-4, help='Description for lr')
    parser.add_argument('--w', type=int, default=256, help='Description for w')
    parser.add_argument('--sigma', type=float, default=0.1, help='Description for sigma')
    parser.add_argument('--model_style', type=str, default='ConditionalFlowMatcher', help='Description for model_style')
    parser.add_argument('--time_varying', type=bool, default=False, help='Description for time_varying')
    parser.add_argument('--resnet_block_groups', type=int, default=4, help='Description for resnet_block_groups')
    parser.add_argument('--learned_sinusoidal_cond', type=bool, default=False, help='Description for learned_sinusoidal_cond')
    parser.add_argument('--learned_sinusoidal_dim', type=int, default=16, help='Description for learned_sinusoidal_dim')
    parser.add_argument('--dim_mults', type=int, nargs='+', default=(1, 2, 4), help='Description for dim_mults')
    parser.add_argument('--model_dim', type=int, default=128, help='Description for model_dim')
    parser.add_argument('--device', type=str, default='cuda:0', help='Description for device')
    parser.add_argument('--epoch_max', type=int, default=10, help='Description for epoch_max')
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