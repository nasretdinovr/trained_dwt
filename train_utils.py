from argparse import ArgumentParser
import time
import torch
import numpy as np

from dataloader import LoadDataset_VCTK as LoadDataset
from denoise import Denoiser
from trainer import Trainer

def parse_agrs():
    parser = ArgumentParser()
    parser.add_argument('--ds_path', type=str, default='./VCTK-Corpus')
    parser.add_argument('--log_dir', type=str, default='./DWT')

    parser.add_argument('--audio_rate', type=int, default=48000)
    parser.add_argument('--random_crop', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--inputLength', type=float, default=2.4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=0.03)

    parser.add_argument('--num_wavelet_levels', type=int, default=6)
    parser.add_argument('--wavelet_size', type=int, default=8)
    parser.add_argument('--wavelet_reg', type=float, default=3e-3)
    parser.add_argument('--net_reg', type=float, default=0)
    parser.add_argument('--wavelet_name', type=str, default='db4')
    parser.add_argument('--trainable_wavelets', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--thresholding_parameter', type=float, default=0.01)
    parser.add_argument('--trainable_threshold', type=bool, default=True)

    parser.add_argument('--thresholding_algorithm', type=str, default='hard')
    parser.add_argument('--threshold_mode', type=str, default='level_dependent')

    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--validation_split', type=float, default=0.1)
    h_params = parser.parse_args()
    return h_params


def train(hparams, eval_only=False):
    torch.manual_seed(1005)
    dataloader_params = {'ds_path': hparams.ds_path,
                     'audio_rate': hparams.audio_rate,
                     'random_crop': hparams.random_crop,
                     'normalize': hparams.normalize,
                     'inputLength': hparams.inputLength,
                     'sigma': hparams.sigma}

    loader = LoadDataset(**dataloader_params)

    dataset_size = len(loader)
    val_size = int(np.floor(hparams.validation_split * dataset_size))
    train_size = dataset_size - val_size

    train_loader, valid_loader = torch.utils.data.random_split(loader, [train_size, val_size])
    dataloader_train = torch.utils.data.DataLoader(train_loader,
                                                   hparams.batch_size,
                                                   shuffle=False,
                                                   pin_memory=False)

    dataloader_test = torch.utils.data.DataLoader(valid_loader,
                                                  hparams.batch_size,
                                                  shuffle=False,
                                                  pin_memory=False)
    now = time.time()
    noised_signal, signal = next(iter(dataloader_train))
    print("Batch uploading time : {}".format(time.time() - now))
    signal_length = noised_signal.size(-1)

    net = Denoiser(hparams.num_wavelet_levels,
                   hparams.wavelet_size,
                   hparams.sigma,
                   thresholding_algorithm=hparams.thresholding_algorithm,
                   threshold_mode=hparams.threshold_mode,
                   signal_length=signal_length,
                   thresholding_parameter=hparams.thresholding_parameter,
                   wavelet_name=hparams.wavelet_name,
                   trainable_wavelets=hparams.trainable_wavelets,
                   trainable_threshold=hparams.trainable_threshold)

    trainer = Trainer(net,
                      hparams.batch_size,
                      log_dir=hparams.log_dir,
                      wavelet_reg=hparams.wavelet_reg,
                      net_reg=hparams.net_reg,
                      lr=hparams.lr)

    if not eval_only:
        trainer.train(dataloader_train, dataloader_test, hparams.epochs)

    mean_val_loss, mean_rmse, mean_snr = trainer.evaluate(dataloader_test)

    metric = {'hparam/rmse': torch.tensor(mean_rmse),
              'hparam/snr': torch.tensor(mean_snr)}
    if hparams.wavelet_name is None:
        hparams.wavelet_name = "None"

    hparams_of_model = vars(hparams)
    trainer.writer.add_hparams(hparams_of_model, metric)
    trainer.writer.close()
    return mean_val_loss, mean_rmse, mean_snr