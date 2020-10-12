from argparse import ArgumentParser
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from source_separation.data.musdb_wrapper.datasets import SingleTrackSet
from source_separation.models.separation_framework import Spectrogram_based
from source_separation.utils.functions import get_activation_by_name, string_to_list
from source_separation.utils.weight_initialization import init_weights_functional


class Dense_UNET(nn.Module):
    @staticmethod
    def get_arg_keys():
        return ['n_blocks',
                'input_channels',
                'internal_channels',
                'n_internal_layers',
                'mk_block_f',
                'mk_ds_f',
                'mk_us_f',
                'first_conv_activation',
                'last_activation',
                't_down_scale',
                'f_down_scale']

    def __init__(self,
                 n_fft,  # framework's
                 n_blocks,
                 input_channels,
                 internal_channels,
                 n_internal_layers,
                 mk_block_f,
                 mk_ds_f,
                 mk_us_f,
                 first_conv_activation,
                 last_activation,
                 t_down_layers,
                 f_down_layers
                 ):

        first_conv_activation = get_activation_by_name(first_conv_activation)
        last_activation = get_activation_by_name(last_activation)

        super(Dense_UNET, self).__init__()

        '''num_block should be an odd integer'''
        assert n_blocks % 2 == 1

        ###########################################################
        # Block-independent Section

        dim_f = n_fft // 2
        input_channels = input_channels

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=internal_channels,
                kernel_size=(1, 2),
                stride=1
            ),
            nn.BatchNorm2d(internal_channels),
            first_conv_activation(),
        )

        self.encoders = nn.ModuleList()
        self.downsamplings = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamplings = nn.ModuleList()

        self.last_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=internal_channels,
                out_channels=input_channels,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1)
            ),
            last_activation()
        )

        self.n = n_blocks // 2

        if t_down_layers is None:
            t_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            t_down_layers = list(range(self.n))
        else:
            t_down_layers = string_to_list(t_down_layers)

        if f_down_layers is None:
            f_down_layers = list(range(self.n))
        elif n_internal_layers == 'None':
            f_down_layers = list(range(self.n))
        else:
            f_down_layers = string_to_list(f_down_layers)

        # Block-independent Section
        ###########################################################

        ###########################################################
        # Block-dependent Section

        f = dim_f

        i = 0
        for i in range(self.n):
            self.encoders.append(mk_block_f(internal_channels, internal_channels, f))
            ds_layer, f = mk_ds_f(internal_channels, i, f, t_down_layers)
            self.downsamplings.append(ds_layer)

        self.mid_block = mk_block_f(internal_channels, internal_channels, f)

        for i in range(self.n):
            us_layer, f = mk_us_f(internal_channels, i, f, self.n, t_down_layers)
            self.upsamplings.append(us_layer)
            self.decoders.append(mk_block_f(2 * internal_channels, internal_channels, f))

        # Block-dependent Section
        ###########################################################

        self.activation = self.last_conv[-1]

    def forward(self, x):

        x = self.first_conv(x)
        encoding_outputs = []

        for i in range(self.n):
            x = self.encoders[i](x)
            encoding_outputs.append(x)
            x = self.downsamplings[i](x)
        x = self.mid_block(x)

        for i in range(self.n):
            x = self.upsamplings[i](x)
            x = torch.cat((x, encoding_outputs[-i - 1]), 1)
            x = self.decoders[i](x)

        return self.last_conv(x)

    def init_weights(self):

        init_weights_functional(self.first_conv, self.first_conv[-1])

        for encoder, downsampling in zip(self.encoders, self.downsamplings):
            encoder.init_weights()
            init_weights_functional(downsampling)

        self.mid_block.init_weights()

        for decoder, upsampling in zip(self.decoders, self.upsamplings):
            decoder.init_weights()
            init_weights_functional(upsampling)

        init_weights_functional(self.last_conv, self.last_conv[-1])


class Dense_UNET_Framework(Spectrogram_based):

    def __init__(self, target_name,
                 n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 spec2spec,
                 optimizer, lr, dev_mode, train_loss, val_loss,
                 layer_level_init_weight,
                 unfreeze_stft_from):

        super(Dense_UNET_Framework, self).__init__(target_name,
                                                   n_fft, hop_length, num_frame,
                                                   spec_type, spec_est_mode, spec2spec,
                                                   optimizer, lr, dev_mode,
                                                   train_loss, val_loss,
                                                   layer_level_init_weight,
                                                   unfreeze_stft_from)

    def to_spec(self, input_signal) -> torch.Tensor:
        if self.magnitude_based:
            return self.stft.to_mag(input_signal).transpose(-1, -3)
        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            return spec_complex.transpose(-1, -3)  # *, 2ch, T, N

    def separate(self, input_signal) -> torch.Tensor:

        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_spec = self.spec2spec(input_spec)

        if self.masking_based:
            output_spec = input_spec * output_spec
        else:
            pass  # Use the original output_spec

        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            # output_spec: *, N, T, 2ch
            output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
            restored = self.stft.restore_complex(output_spec)

        return restored

    def separate_and_return_spec(self, input_signal) -> Tuple[Tensor, Tensor]:

        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_spec = self.spec2spec(input_spec)

        if self.masking_based:
            output_spec = input_spec * output_spec
        else:
            pass  # Use the original output_spec

        output_spec_cache = output_spec
        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            # output_spec: *, N, T, 2ch
            output_spec = output_spec.view(list(output_spec.shape)[:-1] + [2, -1])  # *, N, T, 2, ch
            restored = self.stft.restore_complex(output_spec)

        return restored, output_spec_cache

    def separate_track(self, input_signal) -> torch.Tensor:

        with torch.no_grad():
            db = SingleTrackSet(input_signal, self.hop_length, self.num_frame, self.target_name )
            separated = []

            for item in db:
                separated.append(self.separate(item.unsqueeze(0).to(self.device))[0]
                                 [self.trim_length:-self.trim_length].detach().cpu().numpy())

        import numpy as np
        separated = np.concatenate(separated, axis=0)

        import soundfile
        soundfile.write('temp.wav', separated, 44100)
        return soundfile.read('temp.wav')[0]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_blocks', type=int, default=7)
        parser.add_argument('--input_channels', type=int, default=2)
        parser.add_argument('--internal_channels', type=int, default=24)
        parser.add_argument('--first_conv_activation', type=str, default='relu')
        parser.add_argument('--last_activation', type=str, default='sigmoid')

        parser.add_argument('--t_down_layers', type=tuple, default=None)
        parser.add_argument('--f_down_layers', type=tuple, default=None)

        return Spectrogram_based.add_model_specific_args(parser)
