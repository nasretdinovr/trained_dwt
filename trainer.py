import os
import io
from collections import OrderedDict
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Wavelet_loss import DenoiserLoss
from denoise import rmse_loss, signal_noise_rate
import time
# from graphviz import render
# from torchviz import make_dot, make_dot_from_trace

class Trainer:
    def __init__(self, net, batch_size, log_dir, wavelet_reg, net_reg, lr, optimizer='adam',
                 best_score=float("inf")):
        """
        The classifier used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            batch_size (int): batch size
            wavelet_reg (float): wavelet regularization
            net_reg(float): l2 regularization
            lr (float): learning rate
        """
        self.net = net
        self.epochs = None
        self.global_step = 0
        self.epoch_counter = 0
        self.log_dir = os.path.join('runs', log_dir, 'version_{}'.format(int(time.time())))
        self.batch_size = batch_size
        self.best_score = best_score

        self.wavelet_reg = wavelet_reg
        self.net_reg = net_reg
        self.lr = lr
        self.criterion = DenoiserLoss(wavelet_reg=self.wavelet_reg, net_reg=self.net_reg)

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError('Use "sgd" or "adam" optimizer')

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7,
                                                              patience=0, verbose=True, threshold=0.001)

        if torch.cuda.is_available():
            self.device_train = torch.device('cuda:0')
            self.device_val = torch.device('cuda:0')
        else:
            self.device_train = torch.device('cpu')
            self.device_val = torch.device('cpu')

        self.net.to(self.device_train)

        self.writer = SummaryWriter(self.log_dir)


    def draw_filters_fft(self):
        hi_f = np.abs(fft(self.net.wavelet.hi[0, 0, :].cpu().data.numpy()))
        lo_f = np.abs(fft(self.net.wavelet.lo[0, 0, :].cpu().data.numpy()))
        n = hi_f.shape[-1]
        m = np.max([hi_f.max(), lo_f.max()])
        plt.grid(True)
        plt.tight_layout()
        plt.axis([0, 1, 0, m + m * 0.05])
        plt.plot(np.arange(n // 2 + 1) / (n // 2), lo_f[:n // 2 + 1], 'k--', lw=2)
        plt.plot(np.arange(n // 2 + 1) / (n // 2), hi_f[:n // 2 + 1], 'k', lw=2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.clf()
        return buf

    def _train_epoch(self, train_loader, optimizer, criterion):
        losses = []
        it_count = len(train_loader)
        with tqdm(total=it_count,
                  desc="Epochs {}/{}".format(self.epoch_counter + 1, self.epochs),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for iteration, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device_train), targets.to(self.device_train)
                rmse_before = rmse_loss(targets, inputs)
                snr_before = signal_noise_rate(targets, inputs)

                outputs = self.net(inputs)

                loss, wavelet_loss, mse_loss = criterion(outputs, targets, self.net)

                loss.backward()
                optimizer.step()

                self.writer.add_histogram("wavelet_lo_gradients", self.net.wavelet.lo.grad,
                                          global_step=self.global_step)
                self.writer.add_histogram("wavelet_hi_gradients", self.net.wavelet.hi.grad,
                                          global_step=self.global_step)
                self.writer.add_histogram("threshold_gradients", self.net.threshold.threshold.grad,
                                          global_step=self.global_step)

                optimizer.zero_grad()

                rmse_after = rmse_loss(targets, outputs)
                snr_after = signal_noise_rate(targets, outputs)

                self.writer.add_scalar("loss", loss.item(), global_step=self.global_step)
                self.writer.add_scalar("Wavelet_loss", wavelet_loss.item(), global_step=self.global_step)
                self.writer.add_scalar("MSE_loss", mse_loss.item(), global_step=self.global_step)
                self.writer.add_scalar("RMSE",  rmse_after.item(), global_step=self.global_step)
                self.writer.add_scalar("SNR", snr_after.item(), global_step=self.global_step)



                plot_buf = self.draw_filters_fft()
                image = np.array(Image.open(plot_buf))
                image = np.transpose(image, [2, 0, 1])
                self.writer.add_image("wavelet", image, global_step=self.global_step)

                self.writer.add_histogram("threshold", self.net.threshold.threshold,
                                                global_step=self.global_step)


                self.global_step += 1
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.item()),
                                             wavelet_loss='{0:1.5f}'.format(wavelet_loss.item()),
                                             mse_loss='{0:1.5f}'.format(mse_loss.item()),
                                             snr='{0:1.5f}'.format(snr_after.item())))
                pbar.update(1)

        return loss.item(), wavelet_loss.item(), mse_loss.item()

    def _validate_epoch(self, val_loader, criterion):
        it_count = len(val_loader)
        mean_mse_loss, mean_snr, mean_rmse, rmse_before_mean, snr_before_mean = 0, 0, 0, 0, 0
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device_val), targets.to(self.device_val)

                rmse_before = rmse_loss(targets, inputs)
                snr_before = signal_noise_rate(targets, inputs)

                outputs = self.net(inputs)

                _, _, mse_loss = criterion(outputs, targets, self.net)

                rmse_after = rmse_loss(targets, outputs)
                snr_after = signal_noise_rate(targets, outputs)

                mean_mse_loss += mse_loss.item()
                mean_rmse += rmse_after.item()
                mean_snr += snr_after.item()

                rmse_before_mean += rmse_before.item()
                snr_before_mean += snr_before.item()
                # print(rmse_before_mean, mean_snr)
                # if np.isinf(rmse_before_mean) or np.isinf(mean_snr):
                #     print("Exception")
                #     print("Y the fuck?")
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(mse_loss.item()),
                                             snr='{0:1.5f}'.format(snr_after.item())))
                pbar.update(1)

        self.writer.add_audio('val_noised_audio', inputs[0], sample_rate=48000,
                                  global_step=self.epoch_counter)
        self.writer.add_audio('val_denoised_audio', outputs[0], sample_rate=48000,
                                  global_step=self.epoch_counter)

        self.writer.add_scalar("val_MSE_loss", mean_mse_loss / it_count, global_step=self.global_step)
        self.writer.add_scalars("val_RMSE", {'after': mean_rmse / it_count,
                                             'before': rmse_before_mean / it_count},
                                    global_step=self.global_step)
        self.writer.add_scalars("val_SNR", {'after': mean_snr / it_count,
                                            'before': snr_before_mean / it_count},
                                    global_step=self.global_step)

        return mean_mse_loss / it_count, mean_rmse / it_count, mean_snr / it_count

    def _run_epoch(self, train_loader, val_loader,
                   optimizer, criterion, lr_scheduler):

        # switch to train mode
        self.net.train()
        self.net.to(self.device_train)

        # Run a train pass on the current epoch
        train_loss, wavelet_loss, mse_loss = self._train_epoch(train_loader, optimizer, criterion)

        # switch to evaluate mode
        self.net.eval()
        self.net.to(self.device_val)

        # Run the validation pass
        mean_val_loss, _, _ = self._validate_epoch(val_loader, criterion)

        # Reduce learning rate when needed
        lr_scheduler.step(mean_val_loss, self.epoch_counter)

        if self.best_score > mean_val_loss:
            self.best_score = mean_val_loss
            directory = self.log_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.net, os.path.join(directory, 'best'))
        else:
            directory = self.log_dir
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.net, os.path.join(directory, 'latest'))
        self.epoch_counter += 1

    def evaluate(self, val_loader):
        self.net.eval()
        self.net.to(self.device_val)
        mean_val_loss, mean_rmse, mean_snr = self._validate_epoch(iter(val_loader), self.criterion)
        return mean_val_loss, mean_rmse, mean_snr

    def train(self, train_loader, val_loader, epochs):
        if self.epochs is not None:
            self.epochs += epochs
        else:
            self.epochs = epochs
            self.epoch_counter = 0

        #         self.writer.add_graph(self.net)

        for epoch in range(epochs):
            self._run_epoch(iter(train_loader), iter(val_loader), self.optimizer,
                            self.criterion, self.scheduler)
