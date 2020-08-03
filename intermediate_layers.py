import torch
import torch.nn as nn

class TFC(nn.Module):
    ''' [B, in_channels, T, F] => [B, gr, T, F] '''
    def __init__(self, in_channels, num_layers, gr, kt, kf):
        '''
        in_channels: number of input channels
        num_layers: number of densly connected conv layers
        gr: growth rate
        kt: kernal size of the temporal axis.        
        kf: kernal size of the freq. axis
        '''
        
        super(TFC, self).__init__()
        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1, padding=(kt//2, kf//2)),
                    nn.BatchNorm2d(gr),
                    nn.ReLU(),
                )
            )
            c += gr

    def forward(self, x):
        ''' [B, in_channels, T, F] => [B, gr, T, F] '''
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)  

        return x_
    
class TIF(nn.Module):
    ''' [B, in_channels, T, F] => [B, gr, T, F] '''
    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16):
        
        '''
        channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f 
        bias: bias setting of linear layers
        '''
        
        super(TIF, self).__init__()
        
        if(bn_factor is None):
            self.tif = nn.Sequential(
                nn.Linear(f,f, bias),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
        
        else:
            bn_unis = max(f//bn_factor, min_bn_units)
            self.tif = nn.Sequential(
                nn.Linear(f,bn_unis, bias),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Linear(bn_unis,f,bias),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            
    def forward(self, x):
        return self.tif(x)
        
class TFC_TIF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, bias=False):
        
        '''
        in_channels: number of input channels
        num_layers: number of densly connected conv layers
        gr: growth rate
        kt: kernal size of the temporal axis.        
        kf: kernal size of the freq. axis
        f: num of frequency bins
        
        below are params for TIF 
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f 
        bias: bias setting of linear layers
        '''
        
        super(TFC_TIF, self).__init__()
        self.tfc = TFC(in_channels, num_layers, gr, kt, kf)
        self.tif = TIF(gr, f, bn_factor, bias)
            
    def forward(self, x):
        x = self.tfc(x)
        
        return x + self.tif(x)
    
class TIC(nn.Module):
    ''' [B, in_channels, T, F] => [B, gr, T, F] '''
    def __init__(self, in_channels, num_layers, gr, kf):
        
        '''
        in_channels: number of input channels
        num_layers: number of densly connected conv layers
        gr: growth rate
        kf: kernal size of the freq. axis
        '''
        
        super(TIC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=c, out_channels=gr,kernel_size=kf,stride=1,padding=kf//2),
                    nn.BatchNorm1d(gr),
                    nn.ReLU(),
                )
            )
            c += gr

    def forward(self, x):
        ''' [B, in_channels, T, F] => [B, gr, T, F] '''
        
        B, _, T, F = x.shape
        x = x.transpose(-2,-3) # B, T, c, F
        x = x.reshape(B*T,-1,F)  # BT, c, F
        
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)  

        x_ = x_.view(B,T,-1,F) # B, T, c, F
        x_ = x_.transpose(-2,-3) # B, c, T, F
        return x_
    
       
class TIC_sampling (nn.Module):
    ''' [B, in_channels, T, F] => [B, in_channels, T, F//2 or F*2] '''
    def __init__(self, in_channels, mode='downsampling'):
        
        '''
        in_channels: number of input channels
        '''
        
        super(TIC_sampling, self).__init__()
        self.mode = mode

        if(mode == 'downsampling'):
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        elif(mode == 'upsampling'):
            self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels, kernel_size=2, stride=2)
        else:
            raise NotImplementedError
        self.bn = nn.BatchNorm2d(24)

    def forward(self, x):
        ''' [B, in_channels, T, F] => [B, in_channels, T, F//2] '''
        
        B,C,T,F = x.shape
        
        # B, T, C, F
        x = x.transpose(-2,-3)
        # BT, C, F
        x = x.reshape(-1,C,F)
        # BT, C, F//2 or F*2
        x = self.conv(x)
        # B, T, F//2 or F*2
        x = x.reshape(B,T,C,-1)
        # B, C, T, F//2 or F*2
        x = x.transpose(-2, -3)
        
        return self.bn(x)

class TIF_f1_to_f2(nn.Module):
    ''' [B, in_channels, T, F] => [B, gr, T, F] '''
    def __init__(self, channels, f1, f2, bn_factor=16, bias=False, min_bn_units=16):
        
        '''
        channels:  # channels
        f1: num of frequency bins (input)
        f2: num of frequency bins (output)
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f 
        bias: bias setting of linear layers
        '''
        
        super(TIF_f1_to_f2, self).__init__()
        
        if(bn_factor is None):
            self.tif = nn.Sequential(
                nn.Linear(f1,f2, bias),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
        
        else:
            bn_unis = max(f2//bn_factor, min_bn_units)
            self.tif = nn.Sequential(
                nn.Linear(f1,bn_unis, bias),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Linear(bn_unis,f2,bias),
                nn.BatchNorm2d(channels),
                nn.ReLU()
            )
            
    def forward(self, x):
        return self.tif(x)
    
class TIC_RNN(nn.Module):
    ''' [B, in_channels, T, F] => [B, gr, T, F] '''
    def __init__(self, 
                 in_channels, 
                 num_layers_tic, gr, kf, 
                 f, bn_factor_rnn, num_layers_rnn, bidirectional=True, min_bn_units_rnn=16, bias_rnn=True, 
                 bn_factor_tif=16, bias_tif=True, 
                 skip_connection=True):
        
        '''
        in_channels: number of input channels
        num_layers_tic: number of densly connected conv layers
        gr: growth rate
        kf: kernal size of the freq. axis
        f: # freq bins
        bn_factor_rnn: bottleneck factor of rnn 
        num_layers_rnn: number of layers of rnn
        bidirectional: if true then bidirectional version rnn 
        bn_factor_tif: bottleneck factor of tif
        bias: bias
        skip_connection: if true then tic+rnn else rnn
        '''
        
        super(TIC_RNN, self).__init__()

        self.skip_connection = skip_connection
        
        self.tic = TIC(in_channels, num_layers_tic, gr, kf)
        self.bn = nn.BatchNorm2d(gr)
        
        hidden_units_rnn = max(f//bn_factor_rnn, min_bn_units_rnn)
        self.rnn = nn.GRU(f, hidden_units_rnn, num_layers_rnn, bias=bias_rnn, batch_first=True, bidirectional=bidirectional)
        
        f_from = hidden_units_rnn * 2 if bidirectional else hidden_units_rnn
        f_to = f
        self.tif_f1_to_f2 = TIF_f1_to_f2(gr, f_from, f_to, bn_factor=bn_factor_tif, bias=bias_tif)


    def forward(self, x):
        ''' [B, in_channels, T, F] => [B, gr, T, F] '''
        
        x = self.tic(x) # [B, in_channels, T, F] => [B, gr, T, F]
        x = self.bn(x)  # [B, gr, T, F] => [B, gr, T, F]
        tic_output = x

        B, C, T, F = x.shape
        x = x.view(-1, T, F)
        x, _ = self.rnn(x)       # [B * gr, T, F] => [B * gr, T, 2*hidden_size]
        x = x.view(B,C,T, -1)    # [B * gr, T, 2*hidden_size] => [B, gr, T, 2*hidden_size]
        rnn_output = self.tif_f1_to_f2(x) # [B, gr, T, 2*hidden_size] => [B, gr, T, F]
        
        return tic_output + rnn_output if self.skip_connection else rnn_output
    
class TFC_RNN(nn.Module):
    ''' [B, in_channels, T, F] => [B, gr, T, F] '''

    def __init__(self, in_channels, num_layers_tfc, gr, kt, kf, 
                 f, bn_factor_rnn, num_layers_rnn, bidirectional=True, min_bn_units_rnn=16, bias_rnn=True, 
                 bn_factor_tif=16, bias_tif=True, 
                 skip_connection=True):        
        '''
        in_channels: number of input channels
        num_layers_tfc: number of densly connected conv layers
        gr: growth rate
        kt: kernal size of the temporal axis.        
        kf: kernal size of the freq. axis
        
        f: num of frequency bins
        bn_factor_rnn: bottleneck factor of rnn 
        num_layers_rnn: number of layers of rnn
        bidirectional: if true then bidirectional version rnn 
        bn_factor_tif: bottleneck factor of tif
        bias: bias
        skip_connection: if true then tic+rnn else rnn
        '''
        
        super(TFC_RNN, self).__init__()

        self.skip_connection = skip_connection
        
        self.tfc = TFC(in_channels, num_layers_tfc, gr, kt, kf)
        self.bn = nn.BatchNorm2d(gr)
        
        hidden_units_rnn = max(f//bn_factor_rnn, min_bn_units_rnn)
        self.rnn = nn.GRU(f, hidden_units_rnn, num_layers_rnn, bias=bias_rnn, batch_first=True, bidirectional=bidirectional)
        
        f_from = hidden_units_rnn * 2 if bidirectional else hidden_units_rnn
        f_to = f
        self.tif_f1_to_f2 = TIF_f1_to_f2(gr, f_from, f_to, bn_factor=bn_factor_tif, bias=bias_tif)


    def forward(self, x):
        ''' [B, in_channels, T, F] => [B, gr, T, F] '''
        
        x = self.tfc(x) # [B, in_channels, T, F] => [B, gr, T, F]
        x = self.bn(x)  # [B, gr, T, F] => [B, gr, T, F]
        tfc_output = x

        B, C, T, F = x.shape
        x = x.view(-1, T, F)
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)       # [B * gr, T, F] => [B * gr, T, 2*hidden_size]
        x = x.view(B,C,T, -1)    # [B * gr, T, 2*hidden_size] => [B, gr, T, 2*hidden_size]
        rnn_output = self.tif_f1_to_f2(x) # [B, gr, T, 2*hidden_size] => [B, gr, T, F]
        
        return tfc_output + rnn_output if self.skip_connection else rnn_output