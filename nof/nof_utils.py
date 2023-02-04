"""The functions for some tools
"""

import torch
from torch.optim import SGD, Adam
import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--root_dir', type=str, default='~/ir-mcl/data/ipblab',
                        help='root directory of dataset')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--perturb', type=float, default=0.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='std dev of noise added to regularize sigma')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--L_pos', type=int, default=10,
                        help='the frequency of the positional encoding.')

    # model
    parser.add_argument('--feature_size', type=int, default=256,
                        help='the dimension of the feature maps.')
    parser.add_argument('--use_skip', default=False, action="store_true",
                        help='use skip architecture')

    # maps
    parser.add_argument('--map_size', type=float, default=50,
                        help='the size of the occupancy grid map.')
    parser.add_argument('--map_res', type=float, default=0.05,
                        help='the resolution of the occupancy grid map.')
    parser.add_argument('--lower_origin', default=False, action="store_true",
                        help='use [0, 0] as the origin of the map.')

    # optimization
    parser.add_argument('--seed', type=int, default=None,
                        help='set a seed for fairly comparison during training')

    parser.add_argument('--loss_type', type=str, default='smoothl1',
                        choices=['mse', 'l1', 'smoothl1'],
                        help='loss to use')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lambda_opacity', type=float, default=1e-5,
                        help='the weights of opacity regularization')

    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[200],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(hparams, parameters):
    eps = 1e-8
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr,
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                         weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name) + 1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def decode_batch(batch):
    rays = batch['rays']  # shape: (B, 6)
    rangs = batch['ranges']  # shape: (B, 1)
    return rays, rangs
