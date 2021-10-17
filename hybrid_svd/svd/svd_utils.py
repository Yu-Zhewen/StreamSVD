import numpy as np
import copy
import torch
import torch.nn as nn
from torch import Tensor
import sys

from hybrid_svd.common.utils import *


def generate_low_rank_wrapper(scheme, original_module, original_ofm_size):
    
    if scheme == 0:
        return StaticLowRankScheme0(original_module, original_ofm_size)
    elif scheme == 1:
        return StaticLowRankScheme1(original_module, original_ofm_size)
    elif scheme == 2:
        return StaticLowRankScheme2(original_module, original_ofm_size)
    elif scheme == 3:
        return StaticLowRankScheme3(original_module, original_ofm_size)
    else:
        assert False

class StaticLowRankDecomposition(nn.Module):
    def __init__(self, low_rank_conv1, low_rank_conv2):
        super(StaticLowRankDecomposition, self).__init__()
        self.low_rank_conv1 = low_rank_conv1
        self.low_rank_conv2 = low_rank_conv2

    def forward(self, x: Tensor) -> Tensor:
        intermediate = self.low_rank_conv1(x)
        out = self.low_rank_conv2(intermediate)
        return out


class StaticLowRankDecompositionWrapper():
    def __init__(self, original_module, original_ofm_size):
        self.original_module = original_module
        self.original_ofm_size = original_ofm_size

        assert self.original_module.groups == 1

    def get_original_module_macs(self):
        macs = self.original_ofm_size[0] \
               * self.original_ofm_size[1] \
               * self.original_module.out_channels \
               * self.original_module.in_channels \
               * self.original_module.kernel_size[0] \
               * self.original_module.kernel_size[1]

        return macs

    def get_original_module_params(self):
        params = self.original_module.out_channels \
                 * self.original_module.in_channels \
                 * self.original_module.kernel_size[0] \
                 * self.original_module.kernel_size[1]

        return params

    def decompose_weight(self, unfolded_weight, rank_slice):
        [u, s, vh] = np.linalg.svd(unfolded_weight)

        u_lowrank = u[ : , : , rank_slice]
        s_lowrank = s[ : , rank_slice]
        vh_lowrank = vh[ : , rank_slice, : ]

        u_s_sqrt = np.zeros_like(u_lowrank)
        vh_s_sqrt = np.zeros_like(vh_lowrank)

        for i in range(self.groups):
            s_sqrt_diag = np.diag(np.sqrt(s_lowrank[i]))
            u_s_sqrt[i] = u_lowrank[i] @ s_sqrt_diag
            vh_s_sqrt[i] = s_sqrt_diag @ vh_lowrank[i]

        return u_s_sqrt, vh_s_sqrt
    
    def generate_low_rank_weight(self, rank_slice, quantization_method, weight_quantizer, weight_width, quantise_first, model_name, conv_layer_index):
        if rank_slice == None:
            rank_slice = list(range(self.rank))

        if quantization_method == 0:
            # floating point
            original_weight = self.original_module.weight.detach().clone().numpy()
            unfolded_weight = self.unfold_original_weight(original_weight)
            u_s_sqrt, vh_s_sqrt = self.decompose_weight(unfolded_weight, rank_slice)
            low_rank_weight_array1, low_rank_weight_array2 = self.fold_low_rank_weight(u_s_sqrt, vh_s_sqrt)
        elif quantization_method in [1,3]:
            weight_tensor = self.original_module.weight.detach().clone()
            # fixed point    
            if quantise_first != 0:
                weight_tensor = weight_quantizer.AsymmetricQuantHandler(weight_tensor, self.weight_width, quantization_method)

            original_weight = weight_tensor.numpy()
            unfolded_weight = self.unfold_original_weight(original_weight)
            u_s_sqrt, vh_s_sqrt = self.decompose_weight(unfolded_weight, rank_slice)
            low_rank_weight_array1, low_rank_weight_array2 = self.fold_low_rank_weight(u_s_sqrt, vh_s_sqrt)

            low_rank_weight_tensor1 = weight_quantizer.AsymmetricQuantHandler(torch.from_numpy(low_rank_weight_array1), weight_width, quantization_method)
            low_rank_weight_tensor2 = weight_quantizer.AsymmetricQuantHandler(torch.from_numpy(low_rank_weight_array2), weight_width, quantization_method)
            low_rank_weight_array1 = low_rank_weight_tensor1.detach().numpy()
            low_rank_weight_array2 = low_rank_weight_tensor2.detach().numpy()
        elif quantization_method in [2,4]:
            # iterative quantisation       
            load_from_file = True

            if quantization_method == 2:
                file_prefix = 'checkpoint/iterative_weight/' 
            elif quantization_method == 4:
                file_prefix = 'checkpoint/bfp_iterative_weight/'
            if quantise_first != 0:
                file_prefix += "quantise_first/"
            file_prefix += (model_name + "/w" + str(weight_width))
            iterative_quantized_array1_path = file_prefix + '/subconv1_weight_scheme'+ str(self.scheme)  + '_conv' + str (conv_layer_index) +'.npy'
            iterative_quantized_array2_path = file_prefix + '/subconv2_weight_scheme'+ str(self.scheme)  + '_conv' + str (conv_layer_index) +'.npy'
            
            if load_from_file:
                with open(iterative_quantized_array1_path, "rb") as f:
                    full_rank_weight_array1 = np.load(f)
                with open(iterative_quantized_array2_path, "rb") as f:
                    full_rank_weight_array2 = np.load(f) 
                
                u_s_sqrt, vh_s_sqrt = self.unfold_low_rank_weight(full_rank_weight_array1, full_rank_weight_array2, self.get_max_rank())
                u_s_sqrt = u_s_sqrt[ : , : , rank_slice]
                vh_s_sqrt = vh_s_sqrt[ : , rank_slice, : ]
                low_rank_weight_array1, low_rank_weight_array2 = self.fold_low_rank_weight(u_s_sqrt, vh_s_sqrt)

            else:
                # do not delete
                assert self.rank == self.get_max_rank()

                weight_tensor = self.original_module.weight.detach().clone()
                # fixed point    
                if quantise_first != 0:
                    weight_tensor = weight_quantizer.AsymmetricQuantHandler(weight_tensor, self.weight_width, quantization_method)

                original_weight = weight_tensor.numpy()                
                unfolded_weight = self.unfold_original_weight(original_weight)

                refinement_target = unfolded_weight
                #refinement_target = refinement_target.astype(np.float64)
                
                for r in range(self.rank):
                    u_s_sqrt, vh_s_sqrt = self.decompose_weight(refinement_target, rank_slice)
                    u_s_sqrt = u_s_sqrt[ : , : , 0]
                    vh_s_sqrt = vh_s_sqrt[ : , 0, : ]

                    low_rank_weight_array1, low_rank_weight_array2  = self.fold_low_rank_weight(u_s_sqrt, vh_s_sqrt, 1)
                    low_rank_weight_tensor1 = weight_quantizer.AsymmetricQuantHandler(torch.from_numpy(low_rank_weight_array1), weight_width, quantization_method)
                    low_rank_weight_tensor2 = weight_quantizer.AsymmetricQuantHandler(torch.from_numpy(low_rank_weight_array2), weight_width, quantization_method)
                    low_rank_weight_array1 = low_rank_weight_tensor1.detach().numpy()
                    low_rank_weight_array2 = low_rank_weight_tensor2.detach().numpy()

                    u_s_sqrt, vh_s_sqrt = self.unfold_low_rank_weight(low_rank_weight_array1, low_rank_weight_array2, 1)
                    
                    refinement_target -= u_s_sqrt @ vh_s_sqrt

                    if r == 0:
                        u_s_sqrt_concat = u_s_sqrt
                        vh_s_sqrt_concat = vh_s_sqrt
                    else:
                        u_s_sqrt_concat = np.concatenate((u_s_sqrt_concat, u_s_sqrt), axis = 2)
                        vh_s_sqrt_concat = np.concatenate((vh_s_sqrt_concat, vh_s_sqrt), axis = 1)

                low_rank_weight_array1, low_rank_weight_array2  = self.fold_low_rank_weight(u_s_sqrt_concat, vh_s_sqrt_concat, self.rank)

                with open(iterative_quantized_array1_path, 'wb') as f:
                    np.save(f, low_rank_weight_array1)
                
                with open(iterative_quantized_array2_path, 'wb') as f:
                    np.save(f, low_rank_weight_array2)
        
        else:
            assert False
                
        self.low_rank_conv1.weight.data.copy_(torch.from_numpy(low_rank_weight_array1))
        self.low_rank_conv2.weight.data.copy_(torch.from_numpy(low_rank_weight_array2))
        if self.original_module.bias != None:
            self.low_rank_conv2.bias.data.copy_(self.original_module.bias.detach().clone())

    def export_decomposition(self):
        decomposed_module = StaticLowRankDecomposition(self.low_rank_conv1, self.low_rank_conv2)
        return decomposed_module
                                               
class StaticLowRankScheme0(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme0, self).__init__(original_module, original_ofm_size)
        self.scheme = 0
        self.groups = self.original_module.out_channels

    def get_max_rank(self):
        max_rank = min(self.original_module.in_channels, 
                       self.original_module.kernel_size[0] * self.original_module.kernel_size[1])                          
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * self.original_module.out_channels \
                        * (self.original_module.in_channels * self.original_module.stride[0] * self.original_module.stride[1]
                            + self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.original_module.out_channels \
                            * (self.original_module.in_channels 
                                + self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.reshape(original_weight_array, (self.original_module.out_channels,
                                                                   1, 
                                                                   self.original_module.in_channels,
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = np.transpose(unfolded_weight_array, (0, 2, 1, 3, 4))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (self.original_module.out_channels,
                                                                   self.original_module.in_channels,
                                                                   self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (self.original_module.out_channels,
                                                                       self.original_module.in_channels,
                                                                       1,
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (0, 2, 1, 3, 4))
        approximated_weight_array = np.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))
                    
        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        self.rank = rank
        
        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.original_module.out_channels * self.rank, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.original_module.out_channels * self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=self.original_module.out_channels, 
                                        bias=(self.original_module.bias!=None), dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        low_rank_weight_array1 = np.reshape(u_s_sqrt, (self.original_module.out_channels, 
                                                       self.original_module.in_channels, 
                                                       rank, 
                                                       1, 
                                                       1))
        low_rank_weight_array1 = np.transpose(low_rank_weight_array1, (0, 2, 1, 3, 4))
        low_rank_weight_array1 = np.reshape(low_rank_weight_array1, (self.original_module.out_channels*rank, 
                                                                     self.original_module.in_channels, 
                                                                     1, 
                                                                     1))

        low_rank_weight_array2 = np.reshape(vh_s_sqrt, (self.original_module.out_channels, 
                                                        rank,
                                                        1,
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (0, 2, 1, 3, 4))
        low_rank_weight_array2 = np.reshape(low_rank_weight_array2, (self.original_module.out_channels, 
                                                                     rank,
                                                                     self.original_module.kernel_size[0], 
                                                                     self.original_module.kernel_size[1]))

        return low_rank_weight_array1, low_rank_weight_array2  

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank
        u_s_sqrt = np.reshape(low_rank_weight_array1,(self.original_module.out_channels,
                                                     rank,
                                                     self.original_module.in_channels,
                                                     1,
                                                     1))
        u_s_sqrt = np.transpose(u_s_sqrt, (0, 2, 1, 3, 4))
        u_s_sqrt = np.reshape(u_s_sqrt, (self.original_module.out_channels, 
                                         self.original_module.in_channels, 
                                         rank))

        vh_s_sqrt = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                        1, 
                                                        rank, 
                                                        self.original_module.kernel_size[0],
                                                        self.original_module.kernel_size[1]))  
        vh_s_sqrt = np.transpose(vh_s_sqrt, (0, 2, 1, 3, 4))  
        vh_s_sqrt = np.reshape(vh_s_sqrt, (self.original_module.out_channels, 
                                           rank,
                                           self.original_module.out_channels*self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))    
        return u_s_sqrt, vh_s_sqrt  

class StaticLowRankScheme1(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme1, self).__init__(original_module, original_ofm_size)
        self.scheme = 1
        self.groups = 1

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels, 
                       self.original_module.in_channels * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * (self.original_module.out_channels 
                            + self.original_module.in_channels * self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.original_module.out_channels \
                          + self.original_module.in_channels * self.original_module.kernel_size[0] * self.original_module.kernel_size[1]
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.reshape(original_weight_array, (self.original_module.out_channels,
                                                                   1,
                                                                   self.original_module.in_channels,
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = np.transpose(unfolded_weight_array, (1, 0, 2, 3, 4))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (1,
                                                                   self.original_module.out_channels,
                                                                   self.original_module.in_channels * self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (1,
                                                                       self.original_module.out_channels,
                                                                       self.original_module.in_channels,
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (1, 0, 2, 3, 4))
        approximated_weight_array = np.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        self.rank = rank

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.rank, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        low_rank_weight_array1 = np.reshape(vh_s_sqrt, (rank, 
                                                        self.original_module.in_channels, 
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = np.reshape(u_s_sqrt, (1,
                                                       self.original_module.out_channels,
                                                       rank,
                                                       1, 
                                                       1))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (1, 0, 2, 3, 4))
        low_rank_weight_array2 = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                                     rank,
                                                                     1,
                                                                     1))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank

        vh_s_sqrt = np.reshape(low_rank_weight_array1, (1,
                                                        rank, 
                                                        self.original_module.in_channels*self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))
        u_s_sqrt = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                       1,
                                                       rank,
                                                       1,
                                                       1))
        u_s_sqrt = np.transpose(u_s_sqrt, (1, 0, 2, 3, 4))
        u_s_sqrt = np.reshape(u_s_sqrt, (1,
                                         self.original_module.out_channels,
                                         rank))       

        return u_s_sqrt, vh_s_sqrt

class StaticLowRankScheme2(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme2, self).__init__(original_module, original_ofm_size)
        self.groups = 1
        self.scheme = 2

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels * self.original_module.kernel_size[0], 
                        self.original_module.in_channels * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * (self.original_module.out_channels * self.original_module.kernel_size[0] 
                            + self.original_module.in_channels * self.original_module.kernel_size[1] * self.original_module.stride[0])

        return per_rank_macs

    def get_per_rank_params(self):
        per_rank_params = self.original_module.out_channels \
                            * self.original_module.kernel_size[0] \
                            + self.original_module.in_channels * module.kernel_size[1]
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.transpose(original_weight_array, (0, 2, 1, 3))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (1,
                                                                   self.original_module.out_channels * self.original_module.kernel_size[0],
                                                                   self.original_module.in_channels * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (self.original_module.out_channels,
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.in_channels,
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (0, 2, 1, 3))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        self.rank = rank

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.rank, 
                                        kernel_size=(1, self.original_module.kernel_size[1]), 
                                        stride=(1, self.original_module.stride[1]), 
                                        padding=(0, self.original_module.padding[1]), 
                                        groups=1, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=(self.original_module.kernel_size[0], 1), 
                                        stride=(self.original_module.stride[0], 1), 
                                        padding=(self.original_module.padding[0], 0), 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        low_rank_weight_array1 = np.reshape(vh_s_sqrt, (rank, 
                                                        self.original_module.in_channels, 
                                                        1, 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = np.reshape(u_s_sqrt, (self.original_module.out_channels,
                                                       self.original_module.kernel_size[0], 
                                                       rank, 
                                                       1))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (0, 2 ,1, 3))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank

        vh_s_sqrt = np.reshape(low_rank_weight_array1, (1,
                                                        rank,
                                                        self.original_module.in_channels * self.original_module.kernel_size[1]))
        u_s_sqrt = np.transpose(low_rank_weight_array2, (0, 2 ,1, 3))
        u_s_sqrt = np.reshape(u_s_sqrt, (1,
                                         self.original_module.out_channels * self.original_module.kernel_size[0],
                                         rank))
        return u_s_sqrt, vh_s_sqrt

class StaticLowRankScheme3(StaticLowRankDecompositionWrapper):
    def __init__(self, original_module, original_ofm_size):
        super(StaticLowRankScheme3, self).__init__(original_module, original_ofm_size)
        self.scheme = 3
        self.groups = original_module.in_channels

    def get_max_rank(self):
        max_rank = min(self.original_module.out_channels, 
                       self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return max_rank

    def get_per_rank_macs(self):
        per_rank_macs = self.original_ofm_size[0] \
                        * self.original_ofm_size[1] \
                        * self.original_module.in_channels \
                        * (self.original_module.out_channels 
                            + self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_macs
    
    def get_per_rank_params(self):
        per_rank_params = self.original_module.in_channels \
                            * (self.original_module.out_channels 
                                + self.original_module.kernel_size[0] * self.original_module.kernel_size[1])
        return per_rank_params

    def unfold_original_weight(self, original_weight_array):
        unfolded_weight_array = np.reshape(original_weight_array, (self.original_module.out_channels,
                                                                   self.original_module.in_channels,
                                                                   1,
                                                                   self.original_module.kernel_size[0],
                                                                   self.original_module.kernel_size[1]))
        unfolded_weight_array = np.transpose(unfolded_weight_array, (1, 0, 2, 3, 4))
        unfolded_weight_array = np.reshape(unfolded_weight_array, (self.original_module.in_channels,
                                                                   self.original_module.out_channels,
                                                                   self.original_module.kernel_size[0] * self.original_module.kernel_size[1]))

        return unfolded_weight_array

    def fold_original_weight(self, unfolded_weight_array):
        assert unfolded_weight_array.ndim == 3

        approximated_weight_array = np.reshape(unfolded_weight_array, (self.original_module.in_channels,
                                                                       self.original_module.out_channels,
                                                                       1,
                                                                       self.original_module.kernel_size[0],
                                                                       self.original_module.kernel_size[1]))
        approximated_weight_array = np.transpose(approximated_weight_array, (1, 0, 2, 3, 4))
        approximated_weight_array = np.reshape(approximated_weight_array, (self.original_module.out_channels,
                                                                           self.original_module.in_channels,
                                                                           self.original_module.kernel_size[0],
                                                                           self.original_module.kernel_size[1]))

        return approximated_weight_array

    def initialise_low_rank_module(self, rank):
        self.rank = rank

        self.low_rank_conv1 = nn.Conv2d(self.original_module.in_channels, 
                                        self.original_module.in_channels * self.rank, 
                                        kernel_size=self.original_module.kernel_size, 
                                        stride=self.original_module.stride, 
                                        padding=self.original_module.padding, 
                                        groups=self.original_module.in_channels, 
                                        bias=False, 
                                        dilation=1)

        self.low_rank_conv2 = nn.Conv2d(self.original_module.in_channels * self.rank, 
                                        self.original_module.out_channels, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0, 
                                        groups=1, 
                                        bias=(self.original_module.bias!=None), 
                                        dilation=1)

    def fold_low_rank_weight(self, u_s_sqrt, vh_s_sqrt, rank=None):
        if rank == None:
            rank = self.rank
        low_rank_weight_array1 = np.reshape(vh_s_sqrt, (self.original_module.in_channels * rank, 
                                                        1, 
                                                        self.original_module.kernel_size[0], 
                                                        self.original_module.kernel_size[1]))

        low_rank_weight_array2 = np.reshape(u_s_sqrt, (self.original_module.in_channels,
                                                       self.original_module.out_channels,
                                                       rank,
                                                       1, 
                                                       1))
        low_rank_weight_array2 = np.transpose(low_rank_weight_array2, (1, 0, 2, 3, 4))
        low_rank_weight_array2 = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                                     self.original_module.in_channels * rank,
                                                                     1,
                                                                     1))

        return low_rank_weight_array1, low_rank_weight_array2

    def unfold_low_rank_weight(self, low_rank_weight_array1, low_rank_weight_array2, rank=None):
        if rank == None:
            rank = self.rank

        vh_s_sqrt = np.reshape(low_rank_weight_array1, (self.original_module.in_channels,
                                                        rank, 
                                                        self.original_module.kernel_size[0]*self.original_module.kernel_size[1]))
        u_s_sqrt = np.reshape(low_rank_weight_array2, (self.original_module.out_channels,
                                                       self.original_module.in_channels,
                                                       rank,
                                                       1,
                                                       1))
        u_s_sqrt = np.transpose(u_s_sqrt, (1, 0, 2, 3, 4))
        u_s_sqrt = np.reshape(u_s_sqrt, (self.original_module.in_channels,
                                         self.original_module.out_channels,
                                         rank))       

        return u_s_sqrt, vh_s_sqrt

def svd_target_module_filter(model_name, module):

    if model_name == "mobilenetv2":
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (1, 1):
                return True
        return False

    else:
        if isinstance(module, nn.Conv2d):
            if module.kernel_size == (3, 3) and module.in_channels != 3:
                return True
        return False