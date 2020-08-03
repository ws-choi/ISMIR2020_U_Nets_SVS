import torch
import torch.nn as nn

class U_Net_Framework(nn.Module):
    def __init__(self, io_handler, est_mode, internal_channels, num_blocks, mk_block_f, mk_ds_f, mk_us_f):
        super(U_Net_Framework, self).__init__()
                
        '''num_block should be an odd integer'''
        assert num_blocks % 2 == 1
        
        ## Block-independant Section BEGIN #### Block-independant Section BEGIN #### Block-independant Section BEGIN ##
        
        self.io_handler =io_handler
        dim_f = io_handler.dim_f
        input_channels = io_handler.dim_c
        
        self.est_mode = est_mode
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, 
                out_channels=internal_channels, 
                kernel_size=(2,1), 
                stride=1
            ),
            nn.BatchNorm2d(internal_channels),
            nn.ReLU(),
        )
        
        self.encoders = nn.ModuleList()
        self.downsampling = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsampling = nn.ModuleList()

        self.est_mode = est_mode
        if(est_mode == "cac_mapping"):
            self.final_activation = nn.Identity()
        elif (est_mode == "mag_mapping"):
            self.final_activation = nn.ReLU()
        elif (est_mode == "masking_sigmoid"):
            self.final_activation = nn.Sigmoid()
        elif (est_mode == "masking_tanh"):
            self.final_activation = nn.Tanh()
        elif (est_mode == "masking_relu"):
            self.final_activation = nn.ReLU()            
        else:
            raise NotImplementedError
            
        self.final_conv = nn.Sequential(
            
            nn.Conv2d(
                in_channels=internal_channels, 
                out_channels=input_channels, 
                kernel_size=(2,1), 
                stride=1, 
                padding=(1,0)
            ),
            self.final_activation            
        )
        
        ## Block-independant Section END #### Block-independant Section END #### Block-independant Section END ##

        self.n = num_blocks//2        
        f = dim_f
        
        for i in range(self.n):
            self.encoders.append(mk_block_f(internal_channels, internal_channels, f, i))
            ds_layer, f = mk_ds_f(i, f)
            self.downsampling.append(ds_layer)
        
        self.mid_block = mk_block_f(internal_channels, internal_channels, f, i)
        
        for i in range(self.n):
            
            us_layer, f = mk_us_f(i, f, self.n)
            self.upsampling.append(us_layer)          
            self.decoders.append(mk_block_f(2*internal_channels, internal_channels, f, i))
                    
        
    def forward(self, x):
        
        mix = x
        
        x = self.first_conv(x)
        
        x = x.transpose(-1,-2)
        
        encoding_outputs = []
        
        for i in range(self.n):
            x = self.encoders[i](x)
            encoding_outputs.append(x)
            x = self.downsampling[i](x)
        
        x = self.mid_block(x)
        
        for i in range(self.n):
            x = self.upsampling[i](x)
            x = torch.cat((x, encoding_outputs[-i-1]), 1)
            x = self.decoders[i](x)
        
        x = x.transpose(-1,-2)
        x = self.final_conv(x)

        if(self.est_mode == "cac_mapping" or self.est_mode == "mag_mapping"):
            return x
        else:
            return x * mix
