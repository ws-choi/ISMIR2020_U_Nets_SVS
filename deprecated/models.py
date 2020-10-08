import frameworks
from intermediate_layers import *
import torch
import torch.nn as nn

class TFC_TIF_U_NET (frameworks.U_Net_Framework):
    def __init__(self, spec_loader, est_mode, internal_channels=24, num_blocks=7, num_layers=5, kf=3, kt=3, bias=False, t_scale=None, debug=False):
        
        if(t_scale is None):
            t_scale = range(num_blocks//2)
            
        def mk_block_f (input_c, output_c, f, i, debug=debug):
            if(debug):
                print('intermediate\t at level', i, 'with TFC_TIF')

            return TFC_TIF(in_channels=input_c, num_layers=num_layers, gr=output_c, kf=kf, kt=kt, f=f, bias=True)

        def mk_ds_f (i, f, t_scale=t_scale, debug=debug):
            scale = (2,2) if i in t_scale else (1,2)
            if(debug):
                print('downsampling\t at level', i, 'with scale(T, F): ', scale, ', F_scale: ', f, '->', f//scale[-1])
            ds = nn.Sequential(
                nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return ds, f//scale[-1]

        def mk_us_f (i, f, n, t_scale=t_scale, debug=debug):

            scale = (2,2) if i in [n -1 -s for s in  t_scale] else (1,2)
            if(debug):
                print('upsampling\t at level', i, 'with scale(T, F): ', scale, ', F_scale: ', f, '->', f*scale[-1])
            us = nn.Sequential(
                nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels, kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return us, f*scale[-1]

        super(TFC_TIF_U_NET, self).__init__(spec_loader, est_mode, internal_channels, num_blocks, mk_block_f, mk_ds_f, mk_us_f)