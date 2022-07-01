import math
import torch
import torch.nn as nn
from source_separation.utils.weight_initialization import WI_Module, init_weights_functional


class TFC(WI_Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kt, kf, activation):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kt, kf), stride=1,
                              padding=(kt // 2, kf // 2)),
                    nn.BatchNorm2d(gr),
                    activation(),
                )
            )
            c += gr

        self.activation = self.H[-1][-1]

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)

        return x_


def init_with_domain_knowledge(layer):
    dtype = layer.weight.dtype
    device = layer.weight.device
    with torch.no_grad():
        layer.weight.data = layer.weight.data * 0.0
        in_features = layer.in_features
        out_features = layer.out_features
        bn_factor = out_features / in_features
        scale = math.sqrt(max(math.ceil(bn_factor), math.ceil(1 / bn_factor)))
        if bn_factor < 1:
            layer.weight.data = layer.weight.data + torch.tensor(
                [(1 / scale if math.floor(y * bn_factor) == x else 0) for x in range(out_features) for y in
                 range(in_features)]
                , dtype=dtype
                , device=device
            ).reshape(out_features, in_features)
        else:
            layer.weight.data = layer.weight.data + torch.tensor(
                [(1 / scale if math.floor(x / bn_factor) == y else 0) for x in range(out_features) for y in
                 range(in_features)]
                , dtype=dtype
                , device=device
            ).reshape(out_features, in_features)


class TIF(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU, init_mode=None):

        """
        channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TIF, self).__init__()
        assert init_mode in [None, 'dk']

        self.init_mode = init_mode
        if bn_factor is None:
            self.tif = nn.Sequential(
                nn.Linear(f, f, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation()
            )

        elif bn_factor == 'None' or bn_factor == 'none':
            self.tif = nn.Sequential(
                nn.Linear(f, f, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation()
            )

        else:
            bn_units = max(f // bn_factor, min_bn_units)
            self.bn_units = bn_units
            self.tif = nn.Sequential(
                nn.Linear(f, bn_units, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation(),
                nn.Linear(bn_units, f, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation()
            )

    def forward(self, x):
        return self.tif(x)

    def init_weights(self):
        if self.init_mode is None:
            init_weights_functional(self, self.tif[-1])
        elif self.init_mode == 'dk':  # domain knowledge
            init_weights_functional(self, self.tif[-1])
            init_with_domain_knowledge(self.tif[0])
            if len(self.tif) > 3:
                init_with_domain_knowledge(self.tif[3])
        else:
            raise NotImplementedError


def TDF(channels, f, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU):
    """
    channels: # channels
    f: num of frequency bins
    bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
    bias: bias setting of linear layers
    activation: activation function
    """
    return TIF(channels, f, bn_factor, bias, min_bn_units, activation)


class TFC_TIF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=nn.ReLU, tic_init_mode=None):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        below are params for TIF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers

        activation: activation function
        """

        super(TFC_TIF, self).__init__()
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf, activation)
        self.tif = TIF(gr, f, bn_factor, bias, min_bn_units, activation, tic_init_mode)
        self.activation = self.tif.tif[-1]

    def forward(self, x):
        x = self.tfc(x)
        return x + self.tif(x)

    def init_weights(self):
        self.tfc.init_weights()
        self.tif.init_weights()


def TFC_TDF(in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False, activation=nn.ReLU,
            tic_init_mode=None):
    """
    Wrapper Function: -> TDC_TIF
    in_channels: number of input channels
    num_layers: number of densely connected conv layers
    gr: growth rate
    kt: kernel size of the temporal axis.
    kf: kernel size of the freq. axis
    f: num of frequency bins
    below are params for TDF
    bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
    bias: bias setting of linear layers
    activation: activation
    """
    return TFC_TIF(in_channels, num_layers, gr, kt, kf, f, bn_factor, min_bn_units, bias, activation, tic_init_mode)


class TIC(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kf, activation):

        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kf: kernel size of the freq. axis
        activation: activation
        """

        super(TIC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=c, out_channels=gr, kernel_size=kf, stride=1, padding=kf // 2),
                    nn.BatchNorm1d(gr),
                    activation(),
                )
            )
            c += gr

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """

        B, _, T, F = x.shape
        x = x.transpose(-2, -3)  # B, T, c, F
        x = x.reshape(B * T, -1, F)  # BT, c, F

        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)

        x_ = x_.view(B, T, -1, F)  # B, T, c, F
        x_ = x_.transpose(-2, -3)  # B, c, T, F
        return x_


def TDC(in_channels, num_layers, gr, kf, activation):
    """
    Wrapper Function: -> TIC
    [B, in_channels, T, F] => [B, gr, T, F]
    in_channels: number of input channels
    num_layers: number of densely connected conv layers
    gr: growth rate
    kf: kernel size of the freq. axis
    activation: activation function
    """
    return TIC(in_channels, num_layers, gr, kf, activation)


class TIC_sampling(nn.Module):
    """ [B, in_channels, T, F] => [B, in_channels, T, F//2 or F*2] """

    def __init__(self, in_channels, mode='downsampling'):

        """
        in_channels: number of input channels
        """

        super(TIC_sampling, self).__init__()
        self.mode = mode

        if mode == 'downsampling':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        elif mode == 'upsampling':
            self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        else:
            raise NotImplementedError
        self.bn = nn.BatchNorm2d(24)

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, in_channels, T, F//2] """

        B, C, T, F = x.shape

        # B, T, C, F
        x = x.transpose(-2, -3)
        # BT, C, F
        x = x.reshape(-1, C, F)
        # BT, C, F//2 or F*2
        x = self.conv(x)
        # B, T, F//2 or F*2
        x = x.reshape(B, T, C, -1)
        # B, C, T, F//2 or F*2
        x = x.transpose(-2, -3)

        return self.bn(x)


def TDC_sampling(in_channels, mode='downsampling'):
    """
    wrapper_function: -> TIC_sampling
    [B, in_channels, T, F] => [B, in_channels, T, F//2 or F*2]
    in_channels: number of input channels
    """
    return TIC_sampling(in_channels, mode)


class TIF_f1_to_f2(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f1, f2, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU):

        """
        channels:  # channels
        f1: num of frequency bins (input)
        f2: num of frequency bins (output)
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TIF_f1_to_f2, self).__init__()

        if bn_factor is None:
            self.tif = nn.Sequential(
                nn.Linear(f1, f2, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

        else:
            bn_unis = max(f2 // bn_factor, min_bn_units)
            self.tif = nn.Sequential(
                nn.Linear(f1, bn_unis, bias),
                nn.BatchNorm2d(channels),
                activation(),
                nn.Linear(bn_unis, f2, bias),
                nn.BatchNorm2d(channels),
                activation()
            )

    def forward(self, x):
        return self.tif(x)


class TIC_RNN(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self,
                 in_channels,
                 num_layers_tic, gr, kf,
                 f, bn_factor_rnn, num_layers_rnn, bidirectional=True, min_bn_units_rnn=16, bias_rnn=True,
                 bn_factor_tif=16, bias_tif=True,
                 skip_connection=True,
                 activation=nn.ReLU):
        """
        in_channels: number of input channels
        num_layers_tic: number of densely connected conv layers
        gr: growth rate
        kf: kernel size of the freq. axis
        f: # freq bins
        bn_factor_rnn: bottleneck factor of rnn
        num_layers_rnn: number of layers of rnn
        bidirectional: if true then bidirectional version rnn
        bn_factor_tif: bottleneck factor of tif
        bias: bias
        skip_connection: if true then tic+rnn else rnn
        activation: activation function
        """

        super(TIC_RNN, self).__init__()

        self.skip_connection = skip_connection

        self.tic = TDC(in_channels, num_layers_tic, gr, kf, activation)
        self.bn = nn.BatchNorm2d(gr)

        hidden_units_rnn = max(f // bn_factor_rnn, min_bn_units_rnn)
        self.rnn = nn.GRU(f, hidden_units_rnn, num_layers_rnn, bias=bias_rnn, batch_first=True,
                          bidirectional=bidirectional)

        f_from = hidden_units_rnn * 2 if bidirectional else hidden_units_rnn
        f_to = f
        self.tif_f1_to_f2 = TIF_f1_to_f2(gr, f_from, f_to, bn_factor=bn_factor_tif, bias=bias_tif,
                                         activation=activation)

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """

        x = self.tic(x)  # [B, in_channels, T, F] => [B, gr, T, F]
        x = self.bn(x)  # [B, gr, T, F] => [B, gr, T, F]
        tic_output = x

        B, C, T, F = x.shape
        x = x.view(-1, T, F)
        x, _ = self.rnn(x)  # [B * gr, T, F] => [B * gr, T, 2*hidden_size]
        x = x.view(B, C, T, -1)  # [B * gr, T, 2*hidden_size] => [B, gr, T, 2*hidden_size]
        rnn_output = self.tif_f1_to_f2(x)  # [B, gr, T, 2*hidden_size] => [B, gr, T, F]

        return tic_output + rnn_output if self.skip_connection else rnn_output


class TFC_RNN(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers_tfc, gr, kt, kf,
                 f, bn_factor_rnn, num_layers_rnn, bidirectional=True, min_bn_units_rnn=16, bias_rnn=True,
                 bn_factor_tif=16, bias_tif=True,
                 skip_connection=True,
                 activation=nn.ReLU):
        """
        in_channels: number of input channels
        num_layers_tfc: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis

        f: num of frequency bins
        bn_factor_rnn: bottleneck factor of rnn
        num_layers_rnn: number of layers of rnn
        bidirectional: if true then bidirectional version rnn
        bn_factor_tif: bottleneck factor of tif
        bias: bias
        skip_connection: if true then tic+rnn else rnn

        activation: activation function
        """

        super(TFC_RNN, self).__init__()

        self.skip_connection = skip_connection

        self.tfc = TFC(in_channels, num_layers_tfc, gr, kt, kf, activation)
        self.bn = nn.BatchNorm2d(gr)

        hidden_units_rnn = max(f // bn_factor_rnn, min_bn_units_rnn)
        self.rnn = nn.GRU(f, hidden_units_rnn, num_layers_rnn, bias=bias_rnn, batch_first=True,
                          bidirectional=bidirectional)

        f_from = hidden_units_rnn * 2 if bidirectional else hidden_units_rnn
        f_to = f
        self.tif_f1_to_f2 = TIF_f1_to_f2(gr, f_from, f_to, bn_factor=bn_factor_tif, bias=bias_tif,
                                         activation=activation)

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """

        x = self.tfc(x)  # [B, in_channels, T, F] => [B, gr, T, F]
        x = self.bn(x)  # [B, gr, T, F] => [B, gr, T, F]
        tfc_output = x

        B, C, T, F = x.shape
        x = x.view(-1, T, F)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # [B * gr, T, F] => [B * gr, T, 2*hidden_size]
        x = x.view(B, C, T, -1)  # [B * gr, T, 2*hidden_size] => [B, gr, T, 2*hidden_size]
        rnn_output = self.tif_f1_to_f2(x)  # [B, gr, T, 2*hidden_size] => [B, gr, T, F]

        return tfc_output + rnn_output if self.skip_connection else rnn_output


class u_net_conv_block(WI_Module):
    def __init__(self, bn_layer, conv_layer, activation):
        super(u_net_conv_block, self).__init__()
        self.bn_layer = bn_layer
        self.conv_layer = conv_layer
        self.activation = activation
        self.in_channels = self.conv_layer.conv.in_channels
        self.out_channels = self.conv_layer.conv.out_channels

    def forward(self, x):
        x = self.bn_layer(self.conv_layer(x))
        return self.activation(x)


class u_net_deconv_block(WI_Module):

    def __init__(self, deconv_layer, bn_layer, activation, dropout):
        super(u_net_deconv_block, self).__init__()
        self.bn_layer = bn_layer
        self.deconv_layer = deconv_layer
        self.dropout = nn.Dropout() if dropout else nn.Identity()
        self.activation = activation

    def forward(self, x):
        x = self.bn_layer(self.deconv_layer(x))
        x = self.dropout(x)
        return self.activation(x)


class Conv2d_same(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2)):
        super(Conv2d_same, self).__init__()
        padding = [((k - s + 1) // 2) for k, s in zip(kernel_size, stride)]
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2d_same(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), stride=(2, 2)):
        super(ConvTranspose2d_same, self).__init__()

        # Assuming dilation = 1,
        # H_out = (H_in-1) * stride - 2 * padding + (kernel_size -1) + output_padding + 1
        # We want to make H_out = H_in * stride => Thus,
        # H_in * stride = (H_in-1) * stride - 2 * padding + (kernel_size -1) + output_padding + 1
        # 0 = (0-1) * stride - 2 * padding + kernel_size  + output_padding
        # 2 * padding = -stride + (kernel_size + output_padding)
        # padding = (- stride + kernel_size + output_padding   )/2

        output_padding = [abs(k % 2 - s % 2) for k, s in zip(kernel_size, stride)]
        padding = [(k - s + o) // 2 for k, s, o in zip(kernel_size, stride, output_padding)]
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding
        )

    def forward(self, x):
        return self.deconv(x)
