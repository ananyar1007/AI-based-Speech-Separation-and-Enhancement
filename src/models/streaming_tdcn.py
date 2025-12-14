import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.streaming_tasnet import choose_layer_norm

"""
Time dilated convolutional network.
"""

EPS = 1e-12

class TimeDilatedConvNet(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, num_blocks=3, num_layers=10, dilated=True, separable=False, causal=True,use_batch_norm=False, nonlinear=None, norm=True, eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks

        net = []

        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                net.append(TimeDilatedConvBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, use_batch_norm=use_batch_norm, dual_head=False, eps=eps))
            else:
                net.append(TimeDilatedConvBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, use_batch_norm=use_batch_norm, dual_head=True, eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input):
        num_blocks = self.num_blocks

        x = input
        skip_connection = 0

        for idx in range(num_blocks):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        output = skip_connection

        return output

class TimeDilatedConvBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, num_layers=10, dilated=True, separable=False, causal=True, nonlinear=None, norm=True, use_batch_norm=False, dual_head=True, eps=EPS):
        super().__init__()

        self.num_layers = num_layers

        net = []

        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                print('Dilation', idx, dilation)
                stride = 1
            else:
                dilation = 1
                stride = 2
            if not dual_head and idx == num_layers - 1:
                net.append(ResidualBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, use_batch_norm=use_batch_norm, dual_head=False, eps=eps))
            else:
                net.append(ResidualBlock1d(num_features, hidden_channels=hidden_channels, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm, use_batch_norm=use_batch_norm, dual_head=True, eps=eps))

        self.net = nn.Sequential(*net)

    def forward(self, input):
        num_layers = self.num_layers

        x = input
        skip_connection = 0

        for idx in range(num_layers):
            x, skip = self.net[idx](x)
            skip_connection = skip_connection + skip

        return x, skip_connection

class ResidualBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, skip_channels=256, kernel_size=3, stride=2, dilation=1,
                 separable=False, causal=True, nonlinear=None, norm=True, use_batch_norm=False, dual_head=True, eps=1e-5):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm
        self.dual_head = dual_head

        self.bottleneck_conv1d = nn.Conv1d(num_features, hidden_channels, kernel_size=1, stride=1)

        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
            if use_batch_norm:
                norm_name='BN'
            else:
                norm_name = 'cLN' if causal else 'gLN'
            self.norm1d = choose_layer_norm(norm_name, hidden_channels, causal=causal, eps=eps)
        if separable:
            self.separable_conv1d = DepthwiseSeparableConv1d(hidden_channels, num_features, skip_channels=skip_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, nonlinear=nonlinear, norm=norm, use_batch_norm=use_batch_norm, dual_head=dual_head, eps=eps)
        else:
            if dual_head:
                self.output_conv1d = nn.Conv1d(hidden_channels, num_features, kernel_size=kernel_size, dilation=dilation)
            self.skip_conv1d = nn.Conv1d(hidden_channels, skip_channels, kernel_size=kernel_size, dilation=dilation)
        self.first_time=True

    def forward(self, input):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        separable, causal = self.separable, self.causal
        dual_head = self.dual_head

        _, _, T_original = input.size()

        residual = input
        x = self.bottleneck_conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)
        if norm:
            if x is not None:
                x = self.norm1d(x)
        
#        print('T_original', T_original)
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
#        print('padding', padding)
        if self.first_time:
            if causal:
                padding_left = padding
                padding_right = 0
            else:
                padding_left = padding//2
                padding_right = padding - padding_left
            self.first_time=False
            x = F.pad(x, (padding_left, padding_right))

        if separable:
            output, skip = self.separable_conv1d(x) # output may be None
        else:
            if dual_head:
                output = self.output_conv1d(x)
            else:
                output = None

            skip = self.skip_conv1d(x)
        if output is not None:
            output = output + residual
            return output, skip
        else:
            return None,skip


class StreamingConv1d(nn.Conv1d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        
        self.state=None
        self.state_len = self.kernel_size[0]*self.dilation[0]-self.stride[0]*self.dilation[0]
        print('State size', self.state_len)
    def forward(self, input):
        # input is (1,kL)
        # output is (1,k,N)
      #  print('Encoder:State',self.state.size())
      #  print('Encoder:Input before stat padding',input.size())
        x=None
 #       print('input size',input.size())
        if self.state is None:
            sz_size=0
        else:
            sz_size=self.state.size()[2]
 #           print('Streaming Conv: State size',self.state.size())
            
        if sz_size+input.size()[2] >(self.kernel_size[0]-self.stride[0])*self.dilation[0]:
 #           print('Streaming Conv: input size',input.size())
            if sz_size > 0:
                extended_input = torch.cat([self.state, input], dim=2)
            else:
                extended_input = input
            x = super().forward(extended_input)
         #  print('Encoder:state+input dim', input.size()) 
        # What to do if the input is too short?
        # Add it to state, return empty!
        
        if sz_size > 0:
            self.state = torch.cat([self.state, input[:,:,-(self.kernel_size[0]-self.stride[0])*self.dilation[0]:]], dim=2)
        else:
            self.state =  input[:,:,-(self.kernel_size[0]-self.stride[0])*self.dilation[0]:]
            
# Keep the needed elements
 #       print("State size, before pruning", self.state.size())
        self.state = self.state[:,:,-(self.kernel_size[0]-self.stride[0])*self.dilation[0]:]
 #       print("State size, after pruning", self.state.size())
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=256, skip_channels=256, kernel_size=3, stride=2, dilation=1, causal=True, nonlinear=None, norm=True,use_batch_norm=False, dual_head=True, eps=1e-5):
        super().__init__()
        self.dual_head = dual_head
        self.norm = norm
        self.eps = eps

        self.depthwise_conv1d = StreamingConv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=in_channels)

        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False

        if norm:
             if use_batch_norm:
                 norm_name='BN'
             else:
                 norm_name = 'cLN' if causal else 'gLN'
             self.norm1d = choose_layer_norm(norm_name, in_channels, causal=causal, eps=eps)
        if dual_head:
            self.output_pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)

        self.skip_pointwise_conv1d = nn.Conv1d(in_channels, skip_channels, kernel_size=1, stride=1)

    def forward(self, input):
        nonlinear, norm = self.nonlinear, self.norm
        dual_head = self.dual_head

        x = self.depthwise_conv1d(input)

        if nonlinear:
            x = self.nonlinear1d(x)
        if norm:
            if x is not None:
    #            print('before norm', x.size())
                x = self.norm1d(x)

        if x is None:
            return None,None

        if dual_head:
            output = self.output_pointwise_conv1d(x)
        else:
            output = None

        skip = self.skip_pointwise_conv1d(x)

        return output,skip

def _test_tdcn():
    batch_size = 4
    T = 128
    in_channels, out_channels, skip_channels = 16, 16, 32
    kernel_size = 3
    num_blocks = 3
    num_layers = 4
    dilated, separable = True, False
    causal = True
    nonlinear = 'prelu'
    norm = True

    input = torch.randn((batch_size, in_channels, T), dtype=torch.float)

    model = TimeDilatedConvNet(in_channels, hidden_channels=out_channels, skip_channels=skip_channels, kernel_size=kernel_size, num_blocks=num_blocks, num_layers=num_layers, dilated=dilated, separable=separable, causal=causal, nonlinear=nonlinear, norm=norm)

    print(model)
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    _test_tdcn()
