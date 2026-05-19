import os
import datetime
from collections import OrderedDict

from lib.train.data.wandb_logger import WandbWriter
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from lib.utils.misc import get_world_size

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})
        self.val_log_file = None
        self.loss_curve_file = None
        self.loss_curve_history = []
        self.loader_epoch_summaries = {}

        # Initialize tensorboard and wandb
        self.wandb_writer = None
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval
                self.wandb_writer = WandbWriter(settings.project_path[6:], {}, tensorboard_writer_dir,
                                                cur_train_samples, interval)

            val_log_root = os.path.join(self.settings.save_dir, 'val_log')
            os.makedirs(val_log_root, exist_ok=True)
            val_log_stem = self.settings.project_path.replace('/', '__')
            self.val_log_file = os.path.join(val_log_root, val_log_stem + '.log')
            self.loss_curve_file = os.path.join(val_log_root, val_log_stem + '_loss_curve.png')

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)
            # base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            # print(base_lrs)
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if hasattr(self.actor, "post_backward"):
                        self.actor.post_backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    if hasattr(self.actor, "collect_grad_stats"):
                        grad_stats = self.actor.collect_grad_stats()
                        if grad_stats:
                            stats.update(grad_stats)
                    self.optimizer.step()
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if hasattr(self.actor, "post_backward"):
                        self.actor.post_backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    if hasattr(self.actor, "collect_grad_stats"):
                        grad_stats = self.actor.collect_grad_stats()
                        if grad_stats:
                            stats.update(grad_stats)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            if 'rgb_frames' in data and not isinstance(data['rgb_frames'], list) and data['rgb_frames'] is not None:
                batch_size = data['rgb_frames'].shape[loader.stack_dim]
            elif 'skeleton_frames' in data and not isinstance(data['skeleton_frames'], list) and data['skeleton_frames'] is not None:
                batch_size = data['skeleton_frames'].shape[loader.stack_dim]
            elif 'ir_frames' in data and not isinstance(data['ir_frames'], list) and data['ir_frames'] is not None:
                batch_size = data['ir_frames'].shape[loader.stack_dim]
            elif 'depth_frames' in data and not isinstance(data['depth_frames'], list) and data['depth_frames'] is not None:
                batch_size = data['depth_frames'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics

            self._print_stats(i, loader, batch_size)

            # update wandb status
            if self.wandb_writer is not None and i % self.settings.print_interval == 0:
                if self.settings.local_rank in [-1, 0]:
                    self.wandb_writer.write_log(self.stats, self.epoch)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))

        summary_time = time.time()
        average_fps = self.num_frames / max(summary_time - self.start_time, 1e-8)
        summary = {
            'epoch': self.epoch,
            'loader_name': loader.name,
            'num_batches': loader.__len__(),
            'fps_avg': average_fps,
            'fps_batch': average_fps,
            'data_time': self.avg_date_time / self.num_frames * batch_size,
            'gpu_time': self.avg_gpu_trans_time / self.num_frames * batch_size,
            'forward_time': self.avg_forward_time / self.num_frames * batch_size,
            'total_time': (summary_time - self.start_time) / self.num_frames * batch_size,
        }
        for name, val in self.stats[loader.name].items():
            if hasattr(val, 'avg'):
                summary[name] = val.avg
        self.loader_epoch_summaries[loader.name] = summary

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_val_log()
            self._update_loss_history()
            self._update_loss_curve()
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (
                self.avg_date_time / self.num_frames * batch_size,
                self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        if "Top" in name:
                            print_str += '%s: %.2f  ,  ' % (name, val.avg)
                        else:
                            print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    def _write_val_log(self):
        if self.val_log_file is None:
            return

        for loader in self.loaders:
            if loader.training:
                continue
            summary = self.loader_epoch_summaries.get(loader.name)
            if summary is None:
                continue
            with open(self.val_log_file, 'a', newline='') as f:
                print_str = '[%s: %d, %d / %d] ' % (
                    summary['loader_name'],
                    summary['epoch'],
                    summary['num_batches'],
                    summary['num_batches']
                )
                print_str += 'FPS: %.1f (%.1f)  ,  ' % (summary['fps_avg'], summary['fps_batch'])
                print_str += 'DataTime: %.3f (%.3f)  ,  ' % (summary['data_time'], summary['gpu_time'])
                print_str += 'ForwardTime: %.3f  ,  ' % summary['forward_time']
                print_str += 'TotalTime: %.3f  ,  ' % summary['total_time']
                for name, value in summary.items():
                    if name in ['epoch', 'loader_name', 'num_batches', 'fps_avg', 'fps_batch',
                                'data_time', 'gpu_time', 'forward_time', 'total_time']:
                        continue
                    if "Top" in name:
                        print_str += '%s: %.2f  ,  ' % (name, value)
                    else:
                        print_str += '%s: %.5f  ,  ' % (name, value)
                f.write(print_str[:-5] + '\n')

    def _update_loss_history(self):
        train_summary = self.loader_epoch_summaries.get('train')
        if train_summary is None:
            return

        train_loss = train_summary.get('Loss/total')
        val_name = None
        val_summary = None
        for loader in self.loaders:
            if loader.training:
                continue
            val_name = loader.name
            val_summary = self.loader_epoch_summaries.get(loader.name)
            if val_summary is not None:
                break
        if val_summary is None:
            return

        val_loss = val_summary.get('Loss/total')
        if train_loss is None or val_loss is None:
            return

        self.loss_curve_history.append({
            'epoch': int(self.epoch),
            'train_loss_total': float(train_loss),
            'val_loss_total': float(val_loss),
            'val_loader': val_name,
        })

    def _update_loss_curve(self):
        if self.loss_curve_file is None:
            return
        if not self.loss_curve_history:
            return

        epochs = [row['epoch'] for row in self.loss_curve_history]
        train_losses = [row['train_loss_total'] for row in self.loss_curve_history]
        val_losses = [row['val_loss_total'] for row in self.loss_curve_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, marker='o', linewidth=1.8, label='train_loss')
        plt.plot(epochs, val_losses, marker='o', linewidth=1.8, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/total')
        plt.title(self.settings.project_path.replace('/', ' / '))
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.loss_curve_file, dpi=160)
        plt.close()
