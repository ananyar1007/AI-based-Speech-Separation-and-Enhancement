#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torchaudio
import torch.nn as nn

from utils.utils import set_seed
from dataset import WaveTrainDataset, TrainDataLoader
from driver import Trainer
from models.conv_tasnet import ConvTasNet
from criterion.sdr import NegSISDR
from criterion.pit import PIT1d

parser = argparse.ArgumentParser(description="Training of Conv-TasNet")

parser.add_argument('--wav_root', type=str, default=None, help='Path for dataset ROOT directory')
parser.add_argument('--train_json_path', type=str, default=None, help='Path for train.json')
parser.add_argument('--valid_json_path', type=str, default=None, help='Path for valid.json')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--enc_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase'], help='Encoder type')
parser.add_argument('--dec_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase', 'pinv'], help='Decoder type')
parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--enc_onesided', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns kernel_size // 2 + 1 bins.')
parser.add_argument('--enc_return_complex', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns complex tensor, otherwise real tensor concatenated real and imaginary part in feature dimension.')
parser.add_argument('--n_basis', '-N', type=int, default=512, help='# basis')
parser.add_argument('--kernel_size', '-L', type=int, default=16, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size//2')
parser.add_argument('--sep_bottleneck_channels', '-B', type=int, default=128, help='Bottleneck channels of separator')
parser.add_argument('--sep_hidden_channels', '-H', type=int, default=128, help='Hidden channels of separator')
parser.add_argument('--sep_skip_channels', '-Sc', type=int, default=128, help='Skip connection channels of separator')
parser.add_argument('--sep_kernel_size', '-P', type=int, default=3, help='Skip connection channels of separator')
parser.add_argument('--sep_num_layers', '-X', type=int, default=8, help='# layers of separator')
parser.add_argument('--sep_num_blocks', '-R', type=int, default=3, help='# blocks of separator. Each block has R layers')
parser.add_argument('--dilated', type=int, default=1, help='Dilated convolution')
parser.add_argument('--separable', type=int, default=1, help='Depthwise-separable convolution')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--sep_nonlinear', type=str, default=None, help='Non-linear function of separator')
parser.add_argument('--sep_norm', type=int, default=1, help='Normalization')
parser.add_argument('--use_batch_norm', type=int, default=1, help='BN in seperator')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['sisdr'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default: 0.001')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 128')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    print(args)
    set_seed(args.seed)

    if 1:
        # code for librispeech
        train_dataset = WaveTrainDataset(args.wav_root, args.train_json_path)
        valid_dataset = WaveTrainDataset(args.wav_root, args.valid_json_path)

    
        loader = {}
        loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        loader['valid'] = TrainDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    else:
        train_dataset = torchaudio.datasets.LibriMix(root=args.wav_root, subset= 'train-100', num_speakers = 2, 
                                    sample_rate = 8000, task = 'sep_noisy', mode='max')


        valid_dataset = torchaudio.datasets.LibriMix(root=args.wav_root, subset= 'dev', num_speakers = 2, 
                                    sample_rate = 8000, task = 'sep_noisy', mode='max')

        def collate_fn(batch):
    #        print(type(batch))
    #        print(type(batch[0]))
    #        print(type(batch[0][1]))
    #        print(type(batch[0][2][0]))
            tensor_length=torch.zeros(len(batch))
            for i in range(len(batch)):
                tensor_length[i] = batch[i][1].size()[1]
            l = int(torch.min(tensor_length).item())
     #       print(tensor_length,l)
            y_list=torch.empty(len(batch),1,l)
            z_list=torch.empty(len(batch),2,l)
            for i in range(len(batch)):
    #            print(i)
                y = torch.Tensor(batch[i][1][0,:l])
                z = [torch.Tensor(batch[i][2][0][0,:l]) ,torch.Tensor(batch[i][2][1][0,:l])]
     #           print(y.size())
     #           print(z[0].size(), z[1].size())
                y_list[i,:,:]=y
                z_list[i,0,:]=z[0]
                z_list[i,1,:]=z[1]  #.append(y)
            # z_list.append(z)


            return y_list,z_list
        
        loader = {}
        loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        loader['valid'] = TrainDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

 
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))

    if not args.enc_nonlinear:
        args.enc_nonlinear = None

    model = ConvTasNet(
        args.n_basis, args.kernel_size, stride=args.stride, enc_basis=args.enc_basis, dec_basis=args.dec_basis, enc_nonlinear=args.enc_nonlinear,
        window_fn=args.window_fn, enc_onesided=args.enc_onesided, enc_return_complex=args.enc_return_complex,
        sep_hidden_channels=args.sep_hidden_channels, sep_bottleneck_channels=args.sep_bottleneck_channels, sep_skip_channels=args.sep_skip_channels,
        sep_kernel_size=args.sep_kernel_size, sep_num_blocks=args.sep_num_blocks, sep_num_layers=args.sep_num_layers,
        dilated=args.dilated, separable=args.separable, causal=args.causal, sep_nonlinear=args.sep_nonlinear, sep_norm=args.sep_norm,use_batch_norm=args.use_batch_norm, mask_nonlinear=args.mask_nonlinear,
        n_sources=args.n_sources
    )
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            model = nn.DataParallel(model)
            print("Use CUDA", flush=True)
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA", flush=True)

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))

    # Criterion
    if args.criterion == 'sisdr':
        criterion = NegSISDR()
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    pit_criterion = PIT1d(criterion, n_sources=args.n_sources)

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None

    trainer = Trainer(model, loader, pit_criterion, optimizer,None, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)
    main(args)
