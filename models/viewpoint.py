from os import makedirs
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.networks import ViewAsLinear
from networks.uresnet import Net as Uresnet
from .marrnetbase import MarrnetBaseModel
import math
import pdb
        
# estimate azimuth and elevation
class Model(MarrnetBaseModel):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--pred_depth_minmax',
            action='store_true',
            help="Also predicts depth minmax (for GenRe)",
        )
        return parser, set()

    def __init__(self, opt, logger):
        super(Model, self).__init__(opt, logger)
        self.requires = ['rgb', 'depth', 'silhou', 'normal', 'azimuth', 'elevation']
        if opt.pred_depth_minmax:
            self.requires.append('depth_minmax')
        self.net = Net(
            [3, 1, 1],
            ['normal', 'depth', 'silhou'],
            pred_depth_minmax=opt.pred_depth_minmax,
        )
        self.criterion = nn.functional.mse_loss
        self.kldiv = nn.KLDivLoss()
        self.optimizer = self.adam(
            self.net.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self._nets = [self.net]
        self._optimizers.append(self.optimizer)
        self.input_names = ['rgb']
        self.gt_names = ['depth', 'silhou', 'normal', 'azimuth', 'elevation']
        if opt.pred_depth_minmax:
            self.gt_names.append('depth_minmax')
        self.init_vars(add_path=True)
        self._metrics = ['loss', 'depth', 'silhou', 'normal', 'azimuth', 'elevation']
        if opt.pred_depth_minmax:
            self._metrics.append('depth_minmax')
        self.init_weight(self.net)

    def __str__(self):
        return "MarrNet-3 predicting 2.5D sketches and azimuth/elevation"

    def _train_on_batch(self, epoch, batch_idx, batch):
        self.net.zero_grad()
        pred = self.predict(batch)
        loss, loss_data = self.compute_loss(pred)
        loss.backward()
        self.optimizer.step()
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        pred = self.predict(batch, no_grad=True)
        _, loss_data = self.compute_loss(pred)
        if np.mod(epoch, self.opt.vis_every_vali) == 0:
            if batch_idx < self.opt.vis_batches_vali:
                outdir = join(self.full_logdir, 'epoch%04d_vali' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                self.visualizer.visualize(output, batch_idx, outdir)
                # np.savez(join(outdir, 'batch%04d' % batch_idx), **output)
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    def pack_output(self, pred, batch, add_gt=True):
        pred_normal = pred['normal'].detach().cpu()
        pred_silhou = pred['silhou'].detach().cpu()
        pred_depth = pred['depth'].detach().cpu()
        pred_azimuth = pred['azimuth'].detach().cpu()
        pred_elevation = pred['elevation'].detach().cpu()
        gt_silhou = self.postprocess(batch['silhou'])
        out = {}
        out['rgb_path'] = batch['rgb_path']
        out['pred_normal'] = self.postprocess(pred_normal, bg=1.0, input_mask=gt_silhou).numpy()
        out['pred_silhou'] = self.postprocess(pred_silhou).numpy()
        pred_depth = self.postprocess(pred_depth, bg=0.0, input_mask=gt_silhou)
        out['pred_depth'] = pred_depth.numpy()
        if self.opt.pred_depth_minmax:
            pred_depth_minmax = pred['depth_minmax'].detach()
            pred_abs_depth = self.to_abs_depth(
                (1 - pred_depth).to(torch.device('cuda')),
                pred_depth_minmax
            )  # background is max now
            pred_abs_depth[gt_silhou < 1] = 0  # set background to 0
            out['proj_depth'] = self.proj_depth(pred_abs_depth).cpu().numpy()
            out['pred_depth_minmax'] = pred_depth_minmax.cpu().numpy()
            out['pred_azimuth'] = pred_azimuth.numpy()
            out['pred_elevation'] = pred_elevation.numpy()
        if add_gt:
            out['normal_path'] = batch['normal_path']
            out['silhou_path'] = batch['silhou_path']
            out['depth_path'] = batch['depth_path']
            out['azimuth_path'] = batch['azimuth_path']
            out['elevation_path'] = batch['elevation_path']
            if self.opt.pred_depth_minmax:
                out['gt_depth_minmax'] = batch['depth_minmax'].numpy()
        return out

    def compute_loss(self, pred):
        pred_normal = pred['normal']
        pred_depth = pred['depth']
        pred_silhou = pred['silhou']
        is_fg = self._gt.silhou != 0  # excludes background
        is_fg_full = is_fg.expand_as(pred_normal)
        loss_normal = self.criterion(
            pred_normal[is_fg_full], self._gt.normal[is_fg_full]
        )
        loss_depth = self.criterion(
            pred_depth[is_fg], self._gt.depth[is_fg]
        )
        loss_silhou = self.criterion(pred_silhou, self._gt.silhou)

        w = (256 ** 2) / 2
        loss_azimuth = w * self.kldiv(pred['azimuth'], self._gt.azimuth)
        loss_elevation = w * self.kldiv(pred['elevation'], self._gt.elevation)
        # pdb.set_trace()
        loss = loss_normal + loss_depth + loss_silhou + loss_azimuth + loss_elevation
        loss_data = {}
        loss_data['loss'] = loss.mean().item()
        loss_data['normal'] = loss_normal.mean().item()
        loss_data['depth'] = loss_depth.mean().item()
        loss_data['silhou'] = loss_silhou.mean().item()
        loss_data['azimuth'] = loss_azimuth.mean().item()
        loss_data['elevation'] = loss_elevation.mean().item()

        if self.opt.pred_depth_minmax:
            w_minmax = (256 ** 2) / 2  # matching scale of pixel predictions very roughly
            loss_depth_minmax = w_minmax * self.criterion(
                pred['depth_minmax'],
                self._gt.depth_minmax
            )
            loss += loss_depth_minmax
            loss_data['depth_minmax'] = loss_depth_minmax.mean().item()
        return loss, loss_data


class Net(Uresnet):
    def __init__(self, *args, pred_depth_minmax=True):
        super().__init__(*args)
        self.pred_depth_minmax = pred_depth_minmax
        if self.pred_depth_minmax:
            self.decoder_minmax = nn.Sequential(
                nn.Conv2d(512, 512, 2, stride=2),
                nn.Conv2d(512, 512, 4, stride=1),
                ViewAsLinear(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            )
        self.decoder_azimuth_elevation = nn.Sequential(
            nn.Conv2d(512, 512, 2, stride=2),
            nn.Conv2d(512, 512, 4, stride=1),
            ViewAsLinear(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 24 + 12)
        )

    def forward(self, input_struct):
        x = input_struct.rgb
        out_dict = super().forward(x)
        out = self.decoder_azimuth_elevation(self.encoder_out)
        out_dict['azimuth'] = F.log_softmax(out[..., :24], dim=-1)
        out_dict['elevation'] = F.log_softmax(out[..., 24:], dim=-1)
        if self.pred_depth_minmax:
            out_dict['depth_minmax'] = self.decoder_minmax(self.encoder_out)
        return out_dict
