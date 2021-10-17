from hybrid_svd_submodules import *
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import onnx

from hybrid_svd.svd.svd_optimizer import *

def svd_main(args):

    torch.manual_seed(0)
    random_input = torch.randn(1, 3, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    #load original model
    model = torch.load(args.model_path, map_location=torch.device("cpu"))

    if args.gpu is not None:
        print("Using gpu " + str(args.gpu))
        torch.cuda.set_device(args.gpu)

    if torch.cuda.is_available():
        model = model.cuda()
        random_input = random_input.cuda()

    macs_before_low_rank, params_before_low_rank = calculate_macs_params(model, random_input, False)

    #load dataset
    valdir = os.path.join(args.data, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    svd_optimizer = SvdOptimizer(model, args.model_name, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT), "macs")
    svd_optimizer.model.cpu()
    svd_optimizer.random_input = random_input
    svd_optimizer.val_loader = val_loader

    svd_optimizer.set_quantization_method(args.quantization, args.quantise_first, args.data_width, args.weight_width, 30)

    if args.approximate_scheme == None:
        selected_schemes = svd_optimizer.run_scheme_selection("weight_l2_norm")
    else:
        selected_schemes = args.approximate_scheme

    svd_optimizer.set_schemes(copy.deepcopy(selected_schemes))

    skip_low_rank = True
    for scheme in selected_schemes:
        if scheme != -1:
            skip_low_rank = False

    if args.rank == None and not skip_low_rank:

        svd_optimizer.low_rank_model = copy.deepcopy(svd_optimizer.model)
        if torch.cuda.is_available():
            svd_optimizer.low_rank_model.cuda()

        if args.rank_selection_method == "random":
            iteration_times = 200
            macs_sweep = np.full(iteration_times,-1)
        else:
            if svd_optimizer.model_name == "resnet18":
                sweep_list = [0.9, 0.8, 0.7, 0.6]
            elif svd_optimizer.model_name in ["vgg11_bn","vgg16"]:
                sweep_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            else:
                assert False

            macs_sweep = np.array(sweep_list) * macs_before_low_rank

            if args.rank_selection_method == "rank_taylor":
                svd_optimizer.convert_quantized_low_rank_model(full_rank=True)
                svd_optimizer.gather_taylor_sensitivity(True)

        svd_optimizer.get_rank_selection_scores(args.rank_selection_method)

        for sweep_index, target_macs in enumerate(macs_sweep):
            svd_optimizer.set_schemes(copy.deepcopy(selected_schemes)) 
            sweep_logger = {}
            print("**SWEEP**",target_macs)
            
            svd_optimizer.run_rank_selection_compression_bound("clipped")

            svd_optimizer.convert_quantized_low_rank_model()
            macs_low_rank_step1, _ = calculate_macs_params(svd_optimizer.low_rank_model, random_input, False)
            svd_optimizer.run_rank_selection_compression_bound(args.rank_selection_method, macs_low_rank_step1-target_macs, True)
            print("ranks:", svd_optimizer.selected_ranks)
            
            sweep_logger["schemes"] = svd_optimizer.selected_schemes
            sweep_logger["ranks"] = svd_optimizer.selected_ranks

            svd_optimizer.convert_quantized_low_rank_model(mask_quantization=False)
            macs_before, params_before = calculate_macs_params(svd_optimizer.low_rank_model, random_input, False)
            sweep_logger["macs"] = macs_before
            sweep_logger["params"] = params_before

            if args.data_parallel != None:
                svd_optimizer.low_rank_model = torch.nn.DataParallel(svd_optimizer.low_rank_model.cuda(), args.data_parallel)

            # validate run
            acc1, _ = validate(val_loader, svd_optimizer.low_rank_model, criterion, args.print_freq)
            sweep_logger["acc1"] = acc1.avg.item()

            svd_optimizer.sweep_logger_write_line(sweep_logger, args.output_path + "/sweep_log.csv")
    else:
        sweep_logger = {}
        svd_optimizer.selected_ranks = args.rank

        print("ranks:", svd_optimizer.selected_ranks)
        sweep_logger["schemes"] = svd_optimizer.selected_schemes
        sweep_logger["ranks"] = svd_optimizer.selected_ranks
            
        svd_optimizer.convert_quantized_low_rank_model(mask_quantization=False)
        macs_before, params_before = calculate_macs_params(svd_optimizer.low_rank_model, random_input, False)

        sweep_logger["macs"] = macs_before
        sweep_logger["params"] = params_before

        if args.data_parallel != None:
            svd_optimizer.low_rank_model = torch.nn.DataParallel(svd_optimizer.low_rank_model.cuda(), args.data_parallel)

        # validate run
        acc1, _ = validate(val_loader, svd_optimizer.low_rank_model, criterion, args.print_freq)
        sweep_logger["acc1"] = acc1.avg.item()


        svd_optimizer.sweep_logger_write_line(sweep_logger, args.output_path + "/sweep_log.csv")