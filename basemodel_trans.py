#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:23:57 2021
@author: Boxiang Yun   School:ECNU&HFUT   Email:971950297@qq.com
"""

import torch.nn as nn
import torch
from block_3d import (Decoder,DoubleConv,AdaptivePool_Encoder,
                      Trans_block)
                     

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

    
class EncodeTrans_V3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'cgr' stands for Conv3d+GroupNorm3d+ReLU.
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        choose_translayer(list): use transformer blocks in differenet layers(include encode and decode)
            e.g. [1,1,1,1,0,0,0] all encode layers are choiced
        conv_kernel_size (int or tuple): size of the convolving kernel
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        spatial_size (tuple or list): input image spatial size
        pos_embedway(['random','sincos']): choose different position encode strategy
        max_seq(int): spectral channel size
        use_entmax15('softmax','entamx15','sparsemax','adaptive_entmax'):choose different sparsity strategy
        vis(bool): to visualize attention weight
        
    """
    def __init__(self, in_channels, out_channels, f_maps=64, layer_order='cgr',choose_translayer=[0,1,1,1,0,0,0],tran_enc_layers=1,
                 num_groups=8, num_levels=4, final_sigmoid=True,spatial_size=(256,256),dropout=0.1,pos_embedway='sincos',vis=False,
                 conv_kernel_size=(1,3,3),conv_padding=(0,1,1),max_seq=10, use_entmax15='entmax_bisect', **kwargs):
        super(EncodeTrans_V3DUNet, self).__init__()
        assert len(choose_translayer) == 2*num_levels-1 ,"input correct choiced transformer layers"

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        f_spatials = [[max_seq//2**i, spatial_size[1]//2**i, spatial_size[0]//2**i] for i in range(num_levels)]

        print(f'feature map is {f_maps} and layer_order is {layer_order} and \n feature spatial size is {f_spatials} \n \
              and chooselayer is {choose_translayer} \n conv_kernel_size is {conv_kernel_size} and conv_padding is {conv_padding}')#[32, 64, 128, 256]
        self.vis = vis
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if choose_translayer[i]:
                transf = Trans_block(out_feature_num, spatial_size=f_spatials[i][1:], depth_trans=tran_enc_layers, vis=True,
                                      dropout=dropout,pos_embedway=pos_embedway,use_entmax15=use_entmax15,
                                      seq_length=f_spatials[i][0])
            else: transf = None
            
            if i == 0:
                encoder = AdaptivePool_Encoder(in_channels, out_feature_num,
                                    # skip pooling in the firs encoder
                                      apply_pooling=False,
                                      conv_layer_order=layer_order,
                                      conv_kernel_size=conv_kernel_size,
                                      num_groups=num_groups,
                                      padding=conv_padding,
                                      output_size=f_spatials[i],
                                      transform=transf,
                                      vis=vis,
                                      )
            else:
                encoder = AdaptivePool_Encoder(f_maps[i-1], out_feature_num,
                                      apply_pooling=True,
                                      conv_layer_order=layer_order,
                                      conv_kernel_size=conv_kernel_size,
                                      num_groups=num_groups,
                                      padding=conv_padding,
                                      output_size=f_spatials[i],
                                      transform=transf,
                                      vis=vis)
                       
            encoders.append(encoder)
        reversed_f_maps = list(reversed(f_maps))
        self.encoders = nn.ModuleList(encoders)
        
        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        for i in range(len(reversed_f_maps)-1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            if choose_translayer[i+num_levels]:
                transf = Trans_block(out_feature_num, spatial_size=f_spatials[-i-2][1:], depth_trans=tran_enc_layers, vis=True,
                                      dropout=dropout,pos_embedway=pos_embedway,use_entmax15=use_entmax15,
                                      seq_length=f_spatials[-i-2][0])
            else: transf = None
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=DoubleConv,
                          conv_layer_order='cgr',
                          conv_kernel_size=3,
                          num_groups=num_groups,
                          padding=1,
                          num_spectral=f_spatials[-i-1][0],
                          transform=transf,
                          vis=vis)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
            
    def forward(self, x):
        # encoder part
        encoders_features = []
        atts = []
        #num = len(self.encoders)
        for idx,encoder in enumerate(self.encoders):
            if self.vis:
                x,att = encoder(x)
                if att is not None:
                    atts.append(att)
            else:
                x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        # decoder part
        for idx ,(decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features)):
            if self.vis:
                x,att = decoder(encoder_features,x)
                if att is not None:
                    atts.append(att)
            else:
                x = decoder(encoder_features, x)
        x = self.final_conv(x)
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        x = torch.mean(x,2)
        x = self.final_activation(x)
        if self.vis:
            return x,atts
        else:
            return x
        
if __name__ == '__main__':
    Model = EncodeTrans_V3DUNet(1,1,max_seq=60,num_levels=4,layer_order='cgr',dropout=0.5,
                              f_maps=64,tran_enc_layers=1,pos_embedway='sincos',use_entmax15='adaptive_entmax',
                              spatial_size=(128,128),choose_translayer=[0,1,1,1,0,0,0],
                              conv_kernel_size=(1,3,3),conv_padding=1)
    print(Model)
