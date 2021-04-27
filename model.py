import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from dynamic_conv import Dynamic_conv2d

def conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1,1)):

    out = Dynamic_conv2d(in_planes=in_planes,
                         out_planes=out_planes,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=False)
    return out


class AdaIN(nn.Module):

    def __init__(self, num_features, style_dim, eps=1e-05, momentum=0.1):
        super().__init__()

        self.norm = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=False, track_running_stats=True)
        self.linear_scaling = nn.Linear(style_dim, num_features, bias=True)
        self.linear_shifting = nn.Linear(style_dim, num_features, bias=True)

    # [B, C, D, T], [B, S]
    def forward(self, x, s):
        batch_size = s.size()[0]
        self.gamma = self.linear_scaling(s).view(batch_size, -1, 1, 1)
        self.beta = self.linear_shifting(s).view(batch_size, -1, 1, 1)
        out = self.norm(x)
        out = self.gamma * out + self.beta
        return out


class Identity(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self,x):
        return x

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x*(torch.tanh(F.softplus(x)))

        return x


class SpecNumInputSequential(nn.Sequential):
    def set_num_of_inputs(self, num_of_inputs_list):
        self._num_of_inputs_list=num_of_inputs_list
        assert len(self._modules.values()) == len(self._num_of_inputs_list), \
            'The len of num_of_inputs_list should be the same as num of modules, ' \
            'while their len are %d and %d respectively' % (len(self._num_of_inputs_list),
                                                            len(self._modules.values()))

    def forward(self, *inputs):
        assert len(self._modules.values()) == len(self._num_of_inputs_list), \
            'The len of num_of_inputs_list should be the same as num of modules, ' \
            'while their len are %d and %d respectively' % (len(self._num_of_inputs_list),
                                                            len(self._modules.values()))

        for num_of_inputs, module in zip(self._num_of_inputs_list, self._modules.values()):
            if num_of_inputs==1:
                if type(inputs)==tuple:
                    output = module(inputs[0])
                    inputs = (output,) + inputs[1:]
                else:
                    inputs=module(inputs)
            else:
                assert type(inputs)==tuple
                assert len(inputs)>=num_of_inputs
                inputs = (module(*inputs[:num_of_inputs]),) + inputs[1:]
        if type(inputs)==tuple:
            return inputs[0]
        else:
            return inputs


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            conv2d(dim_in, dim_out),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            Mish()
        )

    def forward(self, x):

        return x+self.main(x)

class ResidualBlockAdaIN(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out, style_num):
        super(ResidualBlockAdaIN, self).__init__()
        self.conv = conv2d(dim_in, dim_out)
        self.ada = AdaIN(dim_out, style_num)
        self.mish = Mish()

    def forward(self, x, s):
        out = self.conv(x)
        out = self.ada(out, s)
        out = self.mish(out)

        return x+out

class ResBlockPreActivation1d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1):
        super(ResBlockPreActivation1d, self).__init__()

        if input_dim != output_dim:
            self.shortcut_proj = nn.Conv1d(input_dim, output_dim, 1)
        else:
            self.shortcut_proj = Identity()
        self.main = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding),
                nn.ReLU(),
                nn.Conv1d(output_dim, output_dim, kernel_size, stride, padding))

    def forward(self, x):
        return self.shortcut_proj(x) + self.main(x)


class StyleEncoder(nn.Module):
    def __init__(self, input_size=(36, 256), ch=64, n_intermediates=5, num_domains=8, dim_style=64):
        super(StyleEncoder, self).__init__()

        self.num_domains = num_domains
        self.dim_style = dim_style

        layers = []

        layers += [nn.Conv1d(input_size[0], ch, 1)]

        for n_intermediate in range(n_intermediates):
            if n_intermediate < n_intermediates - 1:
                layers += [
                    ResBlockPreActivation1d(ch, ch * 2, 3, 1, 1),
                    nn.AvgPool1d(2)
                ]
                ch *= 2
            else:
                # last intermediate resblock doesn't increase filter dimension
                layers += [
                    ResBlockPreActivation1d(ch, ch, 3, 1, 1),
                    nn.AvgPool1d(2)
                ]

        activation_height = input_size[1] // 2 ** n_intermediates  # 4
        layers += [
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(ch, ch, activation_height, 1, 0), # Conv4x4 4->1
            nn.LeakyReLU(negative_slope=0.1),
        ]

        self.conv = nn.Sequential(*layers)

        self.linear = nn.Linear(ch, dim_style)
    def forward(self, x):
        # y = torch.LongTensor([0, 1]).cuda()

        x = x.squeeze(1)

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        out = self.linear(x)
        return out


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=128, style_dim=64, repeat_num=8):
        super(Generator, self).__init__()

        num_of_inputs_list = []
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(Mish())
        num_of_inputs_list += [1, 1, 1]

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(Mish())
            curr_dim = curr_dim * 2
            num_of_inputs_list += [1, 1, 1]

        # Bottleneck layers.
        for i in range(repeat_num // 2):
            layers.append(ResidualBlock(curr_dim, curr_dim))
            num_of_inputs_list += [1]
        for i in range(repeat_num // 2):
            layers.append(ResidualBlockAdaIN(curr_dim, curr_dim, style_dim))
            num_of_inputs_list += [2]

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(AdaIN(curr_dim//2, style_dim=style_dim))
            layers.append(Mish())
            curr_dim = curr_dim // 2
            num_of_inputs_list += [1, 2, 1]

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        num_of_inputs_list += [1]

        self.main = SpecNumInputSequential(*layers)
        self.main.set_num_of_inputs(num_of_inputs_list)

    def forward(self, x, s):
        # s = torch.randn(2,64).cuda()

        return self.main(x, s)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(36, 256), conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(negative_slope=0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num))  # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num))  # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0,
                                  bias=False)  # padding should be 0

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)

        return out_src


if __name__ == '__main__':
    x = torch.randn(2, 1, 36, 512).cuda()
    y = torch.LongTensor([0, 1])


    d = Discriminator()

    print(d)
    #
    from torchsummary import summary
    summary(d.cuda(), (1,36,256), batch_size=2)
