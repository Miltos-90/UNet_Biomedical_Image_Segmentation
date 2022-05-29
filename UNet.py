import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Block(nn.Module):
    ''' One block of Unet.
        Contains 2 repeated 3 x 3 unpadded convolutions, each followed by a ReLU.
    '''
    
    def __init__(self, in_channel, out_channel, kernel_size):
        ''' Initialisation '''

        super().__init__()
        self.conv_1 = nn.Conv2d(in_channel, out_channel, kernel_size)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, kernel_size)
        self.relu   = nn.ReLU()
        
        # Initialise weights on convolutional layers        
        nn.init.normal_(self.conv_1.weight, mean = 0.0, std = self.init_std(in_channel, kernel_size))
        nn.init.normal_(self.conv_1.weight, mean = 0.0, std = self.init_std(out_channel, kernel_size))
    
    
    @staticmethod
    def init_std(channels, kernel_size):
        ''' Computes std for weight initialisation on the convolutional layers'''
        return 2.0 / np.sqrt(channels * kernel_size ** 2)
    
        
    def forward(self, x):
        ''' Forward Phase '''
        
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        
        return x


class Encoder(nn.Module):
    ''' Contractive Part of Unet '''
    
    def __init__(self, channels):
        '''Initialisation'''
        
        super().__init__()
        
        # Make block list
        modules = []
        
        for in_channel, out_channel in zip(channels[:-1], channels[1:]):
            block    = Block(in_channel = in_channel, out_channel = out_channel, kernel_size = 3)
            modules.append(block)
        
        self.blocks    = nn.ModuleList(modules = modules)
        self.max_pol   = nn.MaxPool2d(kernel_size = 2, stride = None)
        self.feat_maps = [] # Feature map of each block to be concatenated with the decoder part
    
    def forward(self, x):
        '''Forward phase'''
        
        for layer_no, layer in enumerate(self.blocks):
        
            # Run block
            x = layer(x)
            
            if not self.is_final_layer(layer_no):
                
                # Store feature maps for the decoder
                self.feat_maps.append(x)
                
                # Perform max pooling operation
                x = self.max_pol(x)
        
        return x
    
    def is_final_layer(self, layer_no):
        return layer_no == len(self.blocks) - 1

    
class Decoder(nn.Module):
    ''' Expansive Part of Unet '''
    
    def __init__(self, channels):
        '''Initialisation'''
        
        super().__init__()
        
        # Make module lists
        up_convs = []
        blocks   = []
        for in_channel, out_channel in zip(channels[:-1], channels[1:]):
            
            # 2x2 Upconvolution
            upconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size = 2, stride = 2)
            up_convs.append(upconv)
            
            # Block (2 convolutions with ReLUs)
            block = Block(in_channel, out_channel, kernel_size = 3)
            blocks.append(block)
    
        # Make modules
        self.upconvs = nn.ModuleList(up_convs)
        self.blocks  = nn.ModuleList(blocks)
        
        
    def forward(self, x, encoded_feat_maps):
        
        for upconv, block in zip(self.upconvs, self.blocks):
            
            # Apply upconvolution
            x = upconv(x)
            
            # Grab corresponding feature map from the encoder
            fts = encoded_feat_maps.pop()

            # Crop it
            fts = self.crop(fts, x.shape[2], x.shape[3])
            
            # Concatenate it to the input
            x = torch.cat([x, fts], dim = 1)
            
            # Perform convs with ReLUs
            x = block(x)
            
        return x
            
    @staticmethod
    def crop(tnsr, new_H, new_W):
        ''' Center crop an input tensor to shape [hew_H, hew_W] '''
        
        # Grab existing size
        _, _, H, W = tnsr.size()

        # Compute one corner of the image
        x1 = int(round( (H - new_H) / 2.))
        y1 = int(round( (W - new_W) / 2.))

        # Compute the other one
        x2 = x1 + new_H
        y2 = y1 + new_W

        return tnsr[:, :, x1:x2, y1:y2]


class Unet(nn.Module):
    ''' Unet class
        As suggested in "U-Net: Convolutional Networks for Biomedical Image Segmentation" (https://arxiv.org/pdf/1505.04597.pdf)
    '''
    
    def __init__(self, channels, no_classes, output_size = None):
        '''Initialisation'''
        
        super().__init__()
        
        self.output_size = output_size
        
        # Initialise encoder
        self.encoder = Encoder(channels)
        
        # Initialise decoder
        dec_channels = list(reversed(channels[1:])) # Flip the channels for the contractive part (and omit the first one)
        self.decoder = Decoder(dec_channels) 
        
        # Initialise final layer
        self.head    = nn.Conv2d(in_channels = channels[1], out_channels = no_classes, kernel_size = 1)
    
    
    def forward(self, x):
        '''Forward Phase'''
        
        x = self.encoder(x)
        x = self.decoder(x, self.encoder.feat_maps)
        x = self.head(x)
        
        # Retain dimensions
        if self.output_size is not None:
            x = F.interpolate(x, self.output_size)
        
        return x
        