import inspect
from argparse import ArgumentParser

import torch
import torch.nn as nn
from source_separation.models.building_blocks import TDF, TDC_sampling
from source_separation.models.loss_functions import get_loss
from source_separation.models.dense_u_net import Dense_UNET_Framework, Dense_UNET
from source_separation.utils import functions


class TDF_NET(Dense_UNET):

    def __init__(self,
                 n_fft,
                 n_blocks, input_channels, internal_channels,
                 first_conv_activation, last_activation,
                 bn_factor, bias, min_bn_units, tdf_activation,
                 t_down_layers, f_down_layers
                 ):

        tdf_activation = functions.get_activation_by_name(tdf_activation)

        def mk_tdf(in_channels, internal_channels, f):

            tdf = TDF(in_channels, f, bn_factor, bias, min_bn_units, tdf_activation)
            if in_channels == internal_channels:
                return tdf
            else:  # to adjust num of channels
                return nn.Sequential(
                    tdf,
                    nn.Conv2d(in_channels, internal_channels, (1, 1)),
                )

        def mk_ds(internal_channels, i, f, t_down_layers):
            ds = TDC_sampling(internal_channels, mode='downsampling')
            return ds, f // 2

        def mk_us(internal_channels, i, f, n, t_down_layers):
            us = TDC_sampling(internal_channels, mode='upsampling')
            return us, f * 2

        super(TDF_NET, self).__init__(n_fft,  # framework's
                                      n_blocks,
                                      input_channels,
                                      internal_channels,
                                      None,
                                      mk_tdf,
                                      mk_ds,
                                      mk_us,
                                      first_conv_activation,
                                      last_activation,
                                      t_down_layers,
                                      f_down_layers)


class TDF_NET_Framework(Dense_UNET_Framework):

    def __init__(self, target_name,
                 n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 optimizer, lr, dev_mode,
                 train_loss, val_loss,
                 layer_level_init_weight,
                 unfreeze_stft_from,
                 **kwargs):
        valid_kwargs = inspect.signature(TDF_NET.__init__).parameters
        tfc_tif_net_kwargs = dict((name, kwargs[name]) for name in valid_kwargs if name in kwargs)
        tfc_tif_net_kwargs['n_fft'] = n_fft
        spec2spec = TDF_NET(**tfc_tif_net_kwargs)

        train_loss_ = get_loss(train_loss, n_fft, hop_length, **kwargs)
        val_loss_ = get_loss(val_loss, n_fft, hop_length, **kwargs)

        super(Dense_UNET_Framework, self).__init__(target_name, n_fft, hop_length, num_frame,
                                                   spec_type, spec_est_mode,
                                                   spec2spec,
                                                   optimizer, lr, dev_mode,
                                                   train_loss_, val_loss_,
                                                   layer_level_init_weight,
                                                   unfreeze_stft_from)

        valid_kwargs = inspect.signature(TDF_NET_Framework.__init__).parameters
        hp = [key for key in valid_kwargs.keys() if key not in ['self', 'kwargs']]
        hp = hp + [key for key in kwargs if not callable(kwargs[key])]
        self.save_hyperparameters(*hp)

    #
    # @staticmethod
    # def get_arg_keys():
    #     return Spectrogram_based.get_arg_keys() + Dense_UNET.get_arg_keys()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--bn_factor', type=int, default=16)
        parser.add_argument('--min_bn_units', type=int, default=16)
        parser.add_argument('--bias', type=bool, default=False)
        parser.add_argument('--tdf_activation', type=str, default='relu')

        return Dense_UNET_Framework.add_model_specific_args(parser)
