"""The functions for NOF pretraining
Code partially borrowed from
https://github.com/kwea123/nerf_pl/blob/master/train.py.
MIT License
Copyright (c) 2020 Quei-An Chen
"""

# pytorch
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from nof.dataset import nof_dataset
from nof.networks import Embedding, NOF
from nof.render import render_rays
from nof.criteria import nof_loss
from nof.criteria.metrics import abs_error, acc_thres, eval_points
from nof.nof_utils import get_opts, get_learning_rate, get_optimizer, decode_batch


class NOFSystem(LightningModule):
    def __init__(self, hparams):
        super(NOFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        # define model
        self.embedding_position = Embedding(in_channels=2, N_freq=self.hparams.L_pos)

        self.nof = NOF(feature_size=self.hparams.feature_size,
                       in_channels_xy=2 + 2 * self.hparams.L_pos * 2,
                       use_skip=self.hparams.use_skip)

        self.loss = nof_loss[self.hparams.loss_type]()

    # datasets
    def prepare_data(self):
        dataset = nof_dataset['ipb2d']
        kwargs = {'root_dir': self.hparams.root_dir}

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, num_workers=4,
                          batch_size=self.hparams.batch_size, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, num_workers=4,
                          batch_size=1, pin_memory=True)

    def forward(self, rays):
        rendered_rays = render_rays(
            model=self.nof, embedding_xy=self.embedding_position, rays=rays,
            N_samples=self.hparams.N_samples, use_disp=self.hparams.use_disp,
            perturb=self.hparams.perturb, noise_std=self.hparams.noise_std,
            chunk=self.hparams.chunk,
        )

        return rendered_rays

    def configure_optimizers(self):
        parameters = []
        parameters += list(self.nof.parameters())

        self.optimizer = get_optimizer(self.hparams, parameters)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.hparams.decay_step,
                                     gamma=self.hparams.decay_gamma)

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        self.log('lr', get_learning_rate(self.optimizer))

        # load data
        rays, gt_ranges = decode_batch(batch)

        # inference
        results = self.forward(rays)

        # compute loss and record
        pred_ranges = results['depth']
        loss = self.loss(pred_ranges, gt_ranges) + \
               self.hparams.lambda_opacity * results['opacity']
        self.log('train/loss', loss)

        with torch.no_grad():
            abs_error_ = abs_error(pred_ranges, gt_ranges)
            acc_thres_ = acc_thres(pred_ranges, gt_ranges)
            self.log('train/avg_error', abs_error_)
            self.log('train/acc_thres', acc_thres_)

        return loss

    def validation_step(self, batch, batch_idx):
        # load data
        rays, gt_ranges = decode_batch(batch)
        rays = rays.squeeze()  # shape: (N_beams, 6)
        gt_ranges = gt_ranges.squeeze()  # shape: (N_beams,)
        valid_mask_gt = batch['valid_mask_gt'].squeeze()  # shape: (N_beams,)

        # inference
        results = self.forward(rays)
        # record
        pred_ranges = results['depth']

        loss = self.loss(pred_ranges, gt_ranges, valid_mask_gt)
        abs_error_ = abs_error(pred_ranges, gt_ranges, valid_mask_gt)
        acc_thres_ = acc_thres(pred_ranges, gt_ranges, valid_mask_gt)

        rays_o, rays_d = rays[:, :2], rays[:, 2:4]
        pred_pts = torch.column_stack((rays_o + rays_d * pred_ranges.unsqueeze(-1),
                                       torch.zeros(pred_ranges.shape[0]).cuda()))
        gt_pts = torch.column_stack((rays_o + rays_d * gt_ranges.unsqueeze(-1),
                                     torch.zeros(gt_ranges.shape[0]).cuda()))
        cd, fscore = eval_points(pred_pts, gt_pts, valid_mask_gt)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/avg_error', abs_error_, prog_bar=True)
        self.log('val/acc_thres', acc_thres_, prog_bar=True)
        self.log('val/cd', cd, prog_bar=True)
        self.log('val/fscore', fscore, prog_bar=True)


if __name__ == '__main__':
    ########################  Step 0: Define Arguments ########################
    hparams = get_opts()

    if hparams.seed:
        seed_everything(hparams.seed, workers=True)

    ########################  Step 1: Model and Trainer ########################
    nof_system = NOFSystem(hparams=hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/avg_error', mode='min', save_top_k=1, filename='best', save_last=True)

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
    )

    # arguments
    print("CUDA is available:", torch.cuda.is_available())
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=1,
                      num_sanity_val_steps=-1,
                      benchmark=True)

    trainer.fit(nof_system)

    print(checkpoint_callback.best_model_path)
