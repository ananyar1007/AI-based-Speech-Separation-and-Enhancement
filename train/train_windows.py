#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import WaveTrainDataset, TrainDataLoader
from driver import Trainer
from models.conv_tasnet import ConvTasNet
from criterion.sdr import NegSISDR
from criterion.pit import PIT1d



def main(args):
    print(args)
    set_seed(args.seed)

    train_dataset = WaveTrainDataset(args.wav_root, args.train_json_path)
    valid_dataset = WaveTrainDataset(args.wav_root, args.valid_json_path)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))

    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = TrainDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

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

    trainer = Trainer(model, loader, pit_criterion, optimizer, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)
    main(args)
