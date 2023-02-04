import argparse
from torch.utils.data import DataLoader

from tqdm import tqdm

from nof.dataset import IPB2DMapping
from nof.networks import Embedding, NOF
from nof.render import render_rays
from nof.nof_utils import load_ckpt, decode_batch
from nof.criteria.metrics import *


def get_opts():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--root_dir', type=str,
                        default='~/ir-mcl/data/ipblab',
                        help='root directory of dataset')

    # data
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--perturb', type=float, default=0.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='std dev of noise added to regularize sigma')
    parser.add_argument('--L_pos', type=int, default=10,
                        help='the frequency of the positional encoding.')

    # model
    parser.add_argument('--feature_size', type=int, default=256,
                        help='the dimension of the feature maps.')
    parser.add_argument('--use_skip', default=False, action="store_true",
                        help='use skip architecture')

    return parser.parse_args()


def error_metrics(pred, gt, rays, valid_mask_gt):
    # ranges to pointclouds
    rays_o, rays_d = rays[:, :2], rays[:, 2:4]
    pred_pts = torch.column_stack((rays_o + rays_d * pred.unsqueeze(-1),
                                   torch.zeros(pred.shape[0])))
    gt_pts = torch.column_stack((rays_o + rays_d * gt.unsqueeze(-1),
                                 torch.zeros(gt.shape[0])))

    abs_error_ = abs_error(pred, gt, valid_mask_gt)
    acc_thres_ = acc_thres(pred, gt, valid_mask_gt)
    cd, fscore = eval_points(pred_pts, gt_pts, valid_mask_gt)

    return abs_error_, acc_thres_, cd, fscore


def summary_errors(errors):
    print("\nError Evaluation")
    errors = torch.Tensor(errors)

    # Mean Errors
    mean_errors = errors.mean(0)

    print(("\t{:>8}" * 4).format("Avg. Error", "Acc", "CD", "F"))
    print(("\t{: 8.2f}" * 4).format(*mean_errors.tolist()))

    print("\n-> Done!")


if __name__ == '__main__':
    ########################  Step 0: Define ########################
    hparams = get_opts()

    use_cuda: bool = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using: ', device)

    ########################  Step 1: Create Dataset and Model ########################
    print("\nLoading Model and Data")
    # define models
    embedding_position = Embedding(in_channels=2, N_freq=hparams.L_pos)

    # Define NOF model
    nof_model = NOF(feature_size=hparams.feature_size,
                    in_channels_xy=2 + 2 * hparams.L_pos * 2,
                    use_skip=hparams.use_skip)

    # loading pretrained weights
    nof_ckpt = hparams.ckpt_path
    load_ckpt(nof_model, nof_ckpt, model_name='nof')
    nof_model.to(device).eval()

    dataset = IPB2DMapping(hparams.root_dir, split='test')
    print("-> Done!")

    ########################  Step 2: Evaluate Synthesis Scan ########################
    errors = []

    dataloader = DataLoader(dataset, shuffle=False, num_workers=4,
                            batch_size=1, pin_memory=True)

    print("\nSynthesis scans with NOF")
    for data in tqdm(dataloader):
        rays, ranges = decode_batch(data)
        rays = rays.squeeze()  # shape: (N_beams, 6)
        ranges = ranges.squeeze()  # shape: (N_beams,)
        valid_mask_gt = data['valid_mask_gt'].squeeze()  # shape: (N_beams,)

        rays = rays.to(device)
        ranges = ranges.to(device)

        with torch.no_grad():
            rendered_rays = render_rays(
                model=nof_model, embedding_xy=embedding_position, rays=rays,
                N_samples=hparams.N_samples, use_disp=hparams.use_disp, perturb=hparams.perturb,
                noise_std=hparams.noise_std, chunk=hparams.chunk
            )

        pred = rendered_rays['depth'].cpu()
        gt = ranges.cpu()
        errors.append(error_metrics(pred, gt, rays.cpu(), valid_mask_gt))

    print("-> Done!")

    # calculate errors
    summary_errors(errors)

