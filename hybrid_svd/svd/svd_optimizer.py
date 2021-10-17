import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import onnx
import os
import sys
import math
import bisect
import csv

from hybrid_svd.svd.svd_utils import *
from hybrid_svd.quantisation.quan_utils import WeightQuantizer, activation_quantization
from hybrid_svd.pruning.pruning_utils import PrunningFineTuner


class SvdOptimizer():
    def __init__(self, model, model_name, input_feature_map_size, optimzation_target):
        self.model = model
        self.model_name = model_name
        self.input_feature_map_size = input_feature_map_size
        self.optimzation_target = optimzation_target
        
        
        self.selected_schemes = None
        self.selected_groups = None
        self.selected_ranks = None
        self.selected_rank_slices = None

    def set_quantization_method(self, quantization_method, quantise_first, data_width, weight_width, acc_width):
        self.quantization_method = quantization_method
        self.quantise_first = quantise_first
        self.data_width = data_width
        self.weight_width = weight_width
        self.acc_width = acc_width

    def set_schemes(self, schemes):
        self.selected_schemes = schemes
        self.low_rank_modules = []

        conv_layer_index = 0
        current_output_feature_map_size = self.input_feature_map_size
        for name, module in self.model.named_modules(): 
            current_input_feature_map_size = current_output_feature_map_size
            current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)
            if svd_target_module_filter(self.model_name,module): 
                scheme = self.selected_schemes[conv_layer_index]

                if scheme == -1:
                    low_rank_module = -1
                else:
                    low_rank_module = generate_low_rank_wrapper(scheme, module, current_output_feature_map_size)
        
                self.low_rank_modules.append(low_rank_module)

                conv_layer_index += 1

    def run_scheme_selection(self, criterion):
        selected_schemes = []
        budget_sweep = list(np.arange(0.3, 0.95, 0.01))
        candidate_schemes = []
        
        current_output_feature_map_size = self.input_feature_map_size
        for name, module in self.model.named_modules():
            scoreboard = [0, 0, 0, 0]

            current_input_feature_map_size = current_output_feature_map_size
            current_output_feature_map_size = update_feature_map_size(name, module, current_input_feature_map_size)

            if svd_target_module_filter(self.model_name,module):
                candidate_schemes = []
                for i in range(len(scoreboard)):
                    candidate_schemes.append(generate_low_rank_wrapper(i, module, current_output_feature_map_size))

                for compression_rate in budget_sweep:
                    optimal_scheme = 0
                    min_error = float('inf')
                    for scheme_wrapper in candidate_schemes:
                        rank_limit = int((scheme_wrapper.get_original_module_macs() * compression_rate) / scheme_wrapper.get_per_rank_macs())
                        rank_slice = list(range(rank_limit))
                        scheme_wrapper.initialise_low_rank_module(rank_limit)
                        original_weight = module.weight.detach().clone().numpy()
                        unfolded_weight = scheme_wrapper.unfold_original_weight(original_weight)
                        u_s_sqrt, vh_s_sqrt = scheme_wrapper.decompose_weight(unfolded_weight, rank_slice)
                        approximated_unfolded_weight = u_s_sqrt @ vh_s_sqrt
                        approximated_weight = scheme_wrapper.fold_original_weight(approximated_unfolded_weight)
                        
                        scheme_error = np.linalg.norm(approximated_weight-original_weight)
                        
                        if scheme_error < min_error:
                            optimal_scheme = scheme_wrapper.scheme
                            min_error = scheme_error

                    scoreboard[optimal_scheme] += 1

                print(scoreboard)
                optimal_scheme = scoreboard.index(max(scoreboard))  
                selected_schemes.append(optimal_scheme)                          

        return selected_schemes                    

    def gather_taylor_sensitivity(self, low_rank_model):
        if low_rank_model:
            taylor_pruner = PrunningFineTuner(None, self.val_loader, self.model_name, self.low_rank_model, 0, self.random_input)
            self.taylor_score_network = taylor_pruner.get_taylor_candidates(-1,[nn.Conv2d])
        else:
            model = copy.deepcopy(self.model)
            if torch.cuda.is_available():
                model.cuda()

            taylor_pruner = PrunningFineTuner(None, self.val_loader, self.model_name, model, 0, self.random_input)
            self.taylor_score_network = taylor_pruner.get_taylor_candidates(-1,[nn.BatchNorm2d])           

    def get_rank_selection_scores(self, criterion):
        self.layer_score = []

        conv_layer_index = 0
        low_rank_layer_index = 0

        for name, module in self.model.named_modules(): 
            if svd_target_module_filter(self.model_name,module): 
                low_rank_module = self.low_rank_modules[conv_layer_index]

                if low_rank_module == -1:
                    self.layer_score.append([])
                else:
                    max_rank = low_rank_module.get_max_rank()
                    per_rank_macs = low_rank_module.get_per_rank_macs()                    

                    if criterion in ["naive_proportion", "random"]:
                        idle_score = np.full(max_rank, -1)

                        self.layer_score.append(idle_score)
                    elif criterion == "global_singular":
                        original_weight = low_rank_module.original_module.weight.detach().clone().numpy()
                        unfolded_weight = low_rank_module.unfold_original_weight(original_weight)
                    
                        [u,s,vh] = np.linalg.svd(unfolded_weight) 
                        s_group_mean = np.mean(s, axis = 0)
                        #s_group_mean_normalised = s_group_mean / np.sum(s_group_mean)
                        s_group_mean_normalised = s_group_mean / np.cumsum(s_group_mean)
                        s_group_mean_normalised_scaled = s_group_mean_normalised / per_rank_macs

                        self.layer_score.append(s_group_mean_normalised_scaled)
                    elif criterion == "rank_taylor":
                        taylor_score_layer = []
                        for taylor_score_item in self.taylor_score_network:
                            if taylor_score_item[0] == low_rank_layer_index:
                                taylor_score_layer.append(taylor_score_item[2])

                        taylor_score_layer = np.array(taylor_score_layer)
                        if low_rank_module.scheme in [0, 3]:
                            taylor_score_layer = taylor_score_layer.reshape(-1, max_rank).sum(axis=0)

                        self.layer_score.append(taylor_score_layer)
                    elif criterion == "reconstructed_l1":
                        reconstructed_error = []
                        original_weight = low_rank_module.original_module.weight.detach().clone().numpy()
                        for i in range(max_rank):
                            rank_slice = list(range(max_rank))
                            rank_slice.pop(i)
                            unfolded_weight = low_rank_module.unfold_original_weight(original_weight)
                            u_s_sqrt, vh_s_sqrt = low_rank_module.decompose_weight(unfolded_weight, rank_slice)
                            approximated_unfolded_weight = u_s_sqrt @ vh_s_sqrt
                            approximated_weight = low_rank_module.fold_original_weight(approximated_unfolded_weight)
                            reconstructed_error.append(np.linalg.norm((original_weight-approximated_weight).flatten(),ord=1))

                        reconstructed_error = np.array(reconstructed_error) /np.linalg.norm((original_weight).flatten(),ord=1)   
                        
                        self.layer_score.append(reconstructed_error)
                    elif criterion == "reconstructed_l2":
                        reconstructed_error = []
                        original_weight = low_rank_module.original_module.weight.detach().clone().numpy()
                        for i in range(max_rank):
                            rank_slice = list(range(max_rank))
                            rank_slice.pop(i)
                            unfolded_weight = low_rank_module.unfold_original_weight(original_weight)
                            u_s_sqrt, vh_s_sqrt = low_rank_module.decompose_weight(unfolded_weight, rank_slice)
                            approximated_unfolded_weight = u_s_sqrt @ vh_s_sqrt
                            approximated_weight = low_rank_module.fold_original_weight(approximated_unfolded_weight)
                            reconstructed_error.append(np.linalg.norm((original_weight-approximated_weight).flatten(),ord=2))

                        reconstructed_error = np.array(reconstructed_error) /np.linalg.norm((original_weight).flatten(),ord=2)   
                    else:
                        assert False

                conv_layer_index += 1
                low_rank_layer_index += 1
            if isinstance(module, nn.Conv2d):   
                low_rank_layer_index += 1
    
    def run_rank_selection_compression_bound(self, criterion, macs_to_removed=0, pre_selected=False, verbose=True):  
        if verbose:
            print('****** Start SVD Rank Selection ******')
        selected_ranks = [] 
        self.selected_rank_slices = None
        

        if criterion == "clipped":
            total_removable_macs = 0
            for low_rank_module in self.low_rank_modules:
                if low_rank_module == -1:
                    selected_ranks.append(-1)
                else:
                    selected_ranks.append(int(low_rank_module.get_original_module_macs()/low_rank_module.get_per_rank_macs()))


        elif criterion == "naive_proportion":
            assert(macs_to_removed > 0)
            assert(self.optimzation_target == "macs")
            total_removable_macs = 0
            
            for i, low_rank_module in enumerate(self.low_rank_modules):
                if low_rank_module != -1:
                    if pre_selected:
                        total_removable_macs += low_rank_module.get_per_rank_macs() * self.selected_ranks[i]
                    else:
                        total_removable_macs += low_rank_module.get_per_rank_macs() * low_rank_module.get_max_rank()

            removed_proportion =  macs_to_removed / total_removable_macs
            for i, low_rank_module in enumerate(self.low_rank_modules):
                if low_rank_module == -1:
                    selected_ranks.append(-1)
                else:
                    if pre_selected:
                        selected_ranks.append(int(self.selected_ranks[i] * (1 - removed_proportion)))
                    else:
                        selected_ranks.append(int(low_rank_module.get_max_rank() * (1 - removed_proportion)))  

        elif criterion in ["random", "global_singular", "rank_taylor", "reconstructed_l1", "reconstructed_l2", "reconstructed_l1_taylor", "reconstructed_l2_taylor"]:
            assert(macs_to_removed > 0)

            conv_layer_index = 0
            rank_counters = {}
            block_candidates = [] 
            group_sizes = []
            for name, module in self.model.named_modules(): 
                if svd_target_module_filter(self.model_name,module):
                    low_rank_module = self.low_rank_modules[conv_layer_index]  
                    if low_rank_module == -1:
                        rank_counters[name] = -1
                        group_sizes.append(0)
                    else:
                    
                        if self.optimzation_target == "macs":
                            group_size = 1 
                            group_sizes.append(group_size)
                        else:
                            assert(False)

                        layer_score = copy.deepcopy(self.layer_score[conv_layer_index])

                        if pre_selected:
                            pre_selected_rank = self.selected_ranks[conv_layer_index]
                            layer_score = layer_score[:pre_selected_rank]
                            rank_counters[name] = pre_selected_rank
                        else:
                            rank_counters[name] = low_rank_module.get_max_rank()

                        candidates = [(name, i+1, x, low_rank_module.get_per_rank_macs(), 1) for i,x in enumerate(layer_score)]

                        if self.selected_schemes[conv_layer_index] in [1, 2]:
                            for candidate_idx, candidate in enumerate(candidates):
                                if candidate_idx % group_size == 0:
                                    block_candidates.append(candidate)
                                else:
                                    block_candidates[-1] = (candidate[0],
                                                            candidate[1],
                                                            candidate[2]+block_candidates[-1][2],
                                                            candidate[3]+block_candidates[-1][3],
                                                            candidate[4]+block_candidates[-1][4])
                        else:
                            block_candidates += candidates

                    conv_layer_index += 1

            
            rank_slice = {}
            for key, value in rank_counters.items(): 
                rank_slice[key] = list(range(value))
            
            deleted_ranks = {}

            for key, _ in rank_counters.items():
                deleted_ranks[key] = []

            if criterion == "random":
                np.random.seed()
                per_layer_candidate = {}

                if self.model_name == "resnet18":
                    lower_bound = 0.4
                elif self.model_name in ["vgg11_bn","vgg16"]:
                    lower_bound = 0.3
                else:
                    assert False

                for candidate in block_candidates:
                    if candidate[0] in per_layer_candidate.keys():
                        per_layer_candidate[candidate[0]] += [candidate]
                    else:
                        per_layer_candidate[candidate[0]] = [candidate]

                for _, layer_candidates in per_layer_candidate.items():
                    upper_bound_rank = len(layer_candidates)
                    lower_bound_rank = int(upper_bound_rank * lower_bound)
                    rank = np.random.choice(list(range(lower_bound_rank, upper_bound_rank)))

                    for candidate in layer_candidates[rank:]:
                        for j in range(candidate[4]):
                            rank_slice[candidate[0]].remove(int(candidate[1]-1-j))
                            deleted_ranks[candidate[0]] += [int(candidate[1]-1-j)]

                        rank_counters[candidate[0]] -= candidate[4] 

            else:
                sorted_candidates = sorted(block_candidates, key=lambda i: i[2]) 
           
                while macs_to_removed > 0 and len(sorted_candidates) > 0:
                    i = 0
                    #while rank_counters[sorted_candidates[i][0]] > sorted_candidates[i][1]:
                    #    i += 1
                    for j in range(sorted_candidates[i][4]):
                        rank_slice[sorted_candidates[i][0]].remove(int(sorted_candidates[i][1]-1-j))
                        deleted_ranks[sorted_candidates[i][0]] += [int(sorted_candidates[i][1]-1-j)]

                    macs_to_removed -= sorted_candidates[i][3]
                    rank_counters[sorted_candidates[i][0]] -= sorted_candidates[i][4]               

                    del sorted_candidates[i]
                

            selected_ranks = list(rank_counters.values())
            deleted_ranks = list(deleted_ranks.values())
            selected_rank_slices = list(rank_slice.values())
 
            self.selected_rank_slices = selected_rank_slices
        else:
            assert False

        if verbose:
            print('****** Rank Selected ******')

        for i, rank in enumerate(selected_ranks):
            if rank == 0:
                selected_ranks[i] = 1
                selected_rank_slices[i] = [0]
                self.selected_rank_slices = selected_rank_slices

        self.selected_ranks = selected_ranks

    def convert_quantized_low_rank_model(self, full_rank=False, mask_quantization=True, simulate_low_rank=1):
        if mask_quantization:
            quantization_method = 0
        else:
            quantization_method = self.quantization_method

        weight_quantizer = WeightQuantizer(self.model)

        self.low_rank_model = copy.deepcopy(self.model)
        conv_layer_index = 0
        replace_dict ={}
        for name, module in self.low_rank_model.named_modules(): 
            if svd_target_module_filter(self.model_name,module):
                low_rank_module = self.low_rank_modules[conv_layer_index]
                    
                if low_rank_module == -1:
                    if quantization_method != 0:
                        quantized_weight = weight_quantizer.AsymmetricQuantHandler(module.weight, self.weight_width, quantization_method)
                        module.weight.data.copy_(quantized_weight)
                else:                   
                    if full_rank or self.selected_ranks[conv_layer_index] == -1:
                        rank_limit = low_rank_module.get_max_rank()
                    else:    
                        rank_limit = self.selected_ranks[conv_layer_index]

                    if self.selected_rank_slices != None:
                        rank_slice = self.selected_rank_slices[conv_layer_index]
                    else:
                        rank_slice = None

                    low_rank_module.initialise_low_rank_module(rank_limit)
                    low_rank_module.generate_low_rank_weight(rank_slice, quantization_method, weight_quantizer, self.weight_width, self.quantise_first, self.model_name, conv_layer_index)

                    if simulate_low_rank == 1:
                        replace_dict[module] = low_rank_module.export_decomposition()
                    else:
                        assert(quantization_method == 0)
                        u_s_sqrt, vh_s_sqrt = low_rank_module.unfold_low_rank_weight(low_rank_module.low_rank_conv1.weight.detach().clone().numpy(), low_rank_module.low_rank_conv2.weight.detach().clone().numpy())
                        approximated_unfolded_weight = u_s_sqrt @ vh_s_sqrt
                        approximated_weight = low_rank_module.fold_original_weight(approximated_unfolded_weight)      
                        module.weight.data.copy_(torch.from_numpy(approximated_weight))

                conv_layer_index += 1   
            
            elif isinstance(module, nn.Conv2d) and quantization_method != 0:
                quantized_weight = weight_quantizer.AsymmetricQuantHandler(module.weight, self.weight_width, quantization_method)
                module.weight.data.copy_(quantized_weight)
            elif isinstance(module, nn.Linear) and quantization_method != 0:
                quantized_weight = weight_quantizer.AsymmetricQuantHandler(module.weight, self.weight_width, quantization_method)
                module.weight.data.copy_(quantized_weight)

        if simulate_low_rank == 1:
            for name, module in self.low_rank_model.named_modules(): 
                for subname, submodule in module.named_children():
                    if submodule in replace_dict.keys():
                        decomposed_conv = replace_dict[submodule]
                        assert(hasattr(module, subname))
                        setattr(module,subname,decomposed_conv)

        if quantization_method != 0:
            self.low_rank_model = activation_quantization(self.low_rank_model, self.data_width, quantization_method, self.val_loader)

        if quantization_method == 0:
            print('****** SVD Full Precision Model Loaded ******')
        elif quantization_method == 1:
            print('****** SVD POST Quantization Finished ******')
        elif quantization_method == 2:
            print('****** SVD ITERATIVE Quantization Finished ******')
        elif quantization_method == 3:
            print('****** SVD Block FP Quantization Finished ******')
        elif quantization_method == 4:
            print('****** SVD ITERATIVE BFP Quantization Finished ******')
        else:
            assert False

        if torch.cuda.is_available():
            self.low_rank_model.cuda()

    def sweep_logger_write_line(self, sweep_logger, csv_path):
        with open(csv_path, mode='a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(list(sweep_logger.values()))

