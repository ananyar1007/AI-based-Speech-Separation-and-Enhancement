#!/usr/bin/env python3
"""Pass input directly to output.

https://github.com/PortAudio/portaudio/blob/master/test/patest_wire.c



 (or float32) works
"""
# python wire_sound_device.py -i 1 -o 3 -c 1 --blocksize 256 --samplerate 8000
import os
import sys
import argparse
import torch
import torchaudio
import sounddevice as sd
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

sys.path.append('C:\\Users\\raghu\\test\\DNSSgit\\DNN-based_source_separation\\src')

#os.chdir("DNNSS/src")
print(os.getcwd())

from models.streaming_conv_tasnet import ConvTasNet as StreamingConvTasNet

state_dict_path = "C:\\Users\\raghu\\test\\DNSSgit\\DNN-based_source_separation\\egs\\tutorials\\conv-tasnet\\morm4_e100_L128_N512_P3_X6_R3\\s222\\model\\best.pth"
model = StreamingConvTasNet(
    n_basis=512, kernel_size=128, stride=None, enc_basis='trainable', dec_basis='trainable',
    sep_hidden_channels=256, sep_bottleneck_channels=128, sep_skip_channels=128, sep_kernel_size=3, 
    sep_num_blocks=3, sep_num_layers=6,
    dilated=True, separable=True,
    sep_nonlinear='prelu', sep_norm=True,use_batch_norm=False, mask_nonlinear='sigmoid',
    causal=True,
    n_sources=2,
    eps=1e-5,
    enc_nonlinear='relu'
)
model_state=torch.load(state_dict_path, map_location=torch.device('cpu'))
model.load_state_dict(model_state['state_dict'])
model.eval()


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')
parser.add_argument(
    '-c', '--channels', type=int, default=2,
    help='number of channels')
parser.add_argument('--dtype', help='audio data type')
parser.add_argument('--samplerate', type=float, help='sampling rate')
parser.add_argument('--blocksize', type=int, help='block size')
parser.add_argument('--latency', type=float, help='latency in seconds')
args = parser.parse_args(remaining)

position = 0
print(position)

def process_data(indata):

    indata=torch.Tensor(indata)
    indata = torch.reshape(indata, (1,1,indata.numel()))
    with torch.no_grad():
        output = model(indata)
    print(indata.size(), output.size())
    output = torch.reshape(output[:,0,:],(output.size()[2],1))
#    print(output.size())
    return output

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    
    print('Data shape',indata.shape, indata.dtype)
    x = process_data(indata)

 #   print(x.size())
 #   x=torch.reshape(x,(x.numel(),1))
    if x.numel() == indata.size:
        outdata[:] = x.numpy()

#    outdata[:] = indata


try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))