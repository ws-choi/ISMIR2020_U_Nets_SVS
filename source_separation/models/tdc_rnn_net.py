import inspect
from argparse import ArgumentParser

from torch import nn

from source_separation.models.building_blocks import TIC_RNN
from source_separation.models.loss_functions import get_loss
from source_separation.models.dense_u_net import Dense_UNET_Framework, Dense_UNET
from source_separation.utils import functions


class TDC_RNN_NET(Dense_UNET):

    def __init__(self,
                 n_fft,
                 n_blocks, input_channels, internal_channels, n_internal_layers,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 kernel_size_f, bn_factor_rnn, num_layers_rnn, min_bn_units_rnn, bias_rnn,
                 bn_factor_tdf, bias_tdf,
                 tdc_rnn_activation,
                 ):

        tdc_rnn_activation = functions.get_activation_by_name(tdc_rnn_activation)

        def mk_tdc_rnn(in_channels, internal_channels, f):
            return TIC_RNN(in_channels,
                           n_internal_layers,
                           internal_channels,
                           kernel_size_f, f,
                           bn_factor_rnn, num_layers_rnn, True,
                           min_bn_units_rnn, bias_rnn,
                           bn_factor_tdf, bias_tdf,
                           skip_connection=True,
                           activation=tdc_rnn_activation)


        def mk_ds(internal_channels, i, f, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in t_down_layers else (1, 2)
            ds = nn.Sequential(
                nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels,
                          kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return ds, f // scale[-1]

        def mk_us(internal_channels, i, f, n, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in [n - 1 - s for s in t_down_layers] else (1, 2)

            us = nn.Sequential(
                nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels,
                                   kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return us, f * scale[-1]

        super(TDC_RNN_NET, self).__init__(n_fft,  # framework's
                                          n_blocks,
                                          input_channels,
                                          internal_channels,
                                          n_internal_layers,
                                          mk_tdc_rnn,
                                          mk_ds,
                                          mk_us,
                                          first_conv_activation,
                                          last_activation,
                                          t_down_layers,
                                          f_down_layers)


class TDC_RNN_NET_Framework(Dense_UNET_Framework):

    def __init__(self, target_name,
                 n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 optimizer, lr, dev_mode,
                 train_loss, val_loss,
                 layer_level_init_weight,
                 unfreeze_stft_from,
                 **kwargs):

        valid_kwargs = inspect.signature(TDC_RNN_NET.__init__).parameters
        tfc_tif_net_kwargs = dict((name, kwargs[name]) for name in valid_kwargs if name in kwargs)
        tfc_tif_net_kwargs['n_fft'] = n_fft
        spec2spec = TDC_RNN_NET(**tfc_tif_net_kwargs)

        train_loss_ = get_loss(train_loss, n_fft, hop_length, **kwargs)
        val_loss_ = get_loss(val_loss, n_fft, hop_length, **kwargs)

        super(Dense_UNET_Framework, self).__init__(target_name, n_fft, hop_length, num_frame,
                                                   spec_type, spec_est_mode,
                                                   spec2spec,
                                                   optimizer, lr, dev_mode,
                                                   train_loss_, val_loss_,
                                                   layer_level_init_weight,
                                                   unfreeze_stft_from)

        valid_kwargs = inspect.signature(TDC_RNN_NET_Framework.__init__).parameters
        hp = [key for key in valid_kwargs.keys() if key not in ['self', 'kwargs'] ]
        hp = hp + [key for key in kwargs if not callable(kwargs[key])]
        self.save_hyperparameters(*hp)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_internal_layers', type=int, default=5)


        parser.add_argument('--kernel_size_f', type=int, default=3)

        parser.add_argument('--bn_factor_rnn', type=int, default=16)
        parser.add_argument('--bn_factor_tdf', type=int, default=16)
        parser.add_argument('--num_layers_rnn', type=int, default=1)

        parser.add_argument('--bias_rnn', type=bool, default=False)
        parser.add_argument('--bias_tdf', type=bool, default=False)
        parser.add_argument('--min_bn_units_rnn', type=int, default=16)
        parser.add_argument('--tdc_rnn_activation', type=str, default='relu')


        return Dense_UNET_Framework.add_model_specific_args(parser)
