import torch
import torch.nn as nn
import copy

def linear_quantize(x, scaling_factor, zero_point):
    if len(x.shape) == 4:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(x.shape) == 2:
        scaling_factor = scaling_factor.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        assert False

    x_quan = torch.round(scaling_factor * x - zero_point)

    return x_quan


def linear_dequantize(x_quan, scaling_factor, zero_point):
    if len(x_quan.shape) == 4:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(x_quan.shape) == 2:
        scaling_factor = scaling_factor.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        assert False

    x = (x_quan + zero_point) / scaling_factor

    return x 


def asymmetric_linear_no_clipping(wordlength, x_min, x_max):

    scaling_factor = (2**wordlength - 1) / torch.clamp((x_max - x_min), min=1e-8)
    zero_point = scaling_factor * x_min

    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.round()
    else:
        zero_point = float(round(zero_point))

    zero_point += 2**(wordlength - 1)

    return scaling_factor, zero_point

def saturate(w_quan, wordlength):
    n = 2**(wordlength - 1)
    w_quan = torch.clamp(w_quan, -n, n - 1)

    return w_quan

class WeightQuantizer():
    def __init__(self, model):
        bFirst = True

        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if bFirst:
                    bFirst = False
                    self.w_min = torch.min(module.weight)
                    self.w_max = torch.max(module.weight)
                else:
                    self.w_min = torch.minimum(self.w_min, torch.min(module.weight))
                    self.w_max = torch.maximum(self.w_max, torch.max(module.weight))
    
        print("weight min:", self.w_min)
        print("weight max:", self.w_max)

    def AsymmetricQuantHandler(self, w, wordlength, quantization_method):
        weight_size = list(w.size())

        if quantization_method in [3,4]:
            w_block = w.data.contiguous().view(weight_size[0], -1)
            w_min = w_block.min(dim=1).values
            w_max = w_block.max(dim=1).values
        
        elif quantization_method in [1,2]:
            w_min = torch.tensor([self.w_min])
            w_max = torch.tensor([self.w_max])
            w_min = w_min.to(w.device)
            w_max = w_max.to(w.device)
        
        else:
            assert False

        scaling_factor, zero_point = asymmetric_linear_no_clipping(wordlength, w_min, w_max)
        w_quan = linear_quantize(w, scaling_factor, zero_point)
        w_quan = saturate(w_quan, wordlength)         
        w_approx = linear_dequantize(w_quan, scaling_factor, zero_point)

        return w_approx


class QuanAct(nn.Module):
    def __init__(self,
                 activation_wordlength,
                 quantization_method):

        super(QuanAct, self).__init__()
        self.activation_wordlength = activation_wordlength
        self.quantization_method = quantization_method
        self.gather_data = True
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('scaling_factor', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def get_scale_shift(self):
        self.scaling_factor, self.zero_point = asymmetric_linear_no_clipping(self.activation_wordlength, self.x_min, self.x_max)

    def forward(self, x):

        if self.gather_data:
            x_min = x.data.min()
            x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max += -self.x_max + max(self.x_max, x_max) 

            return x           
        else:
            x_quan = linear_quantize(x, self.scaling_factor, self.zero_point)
            x_quan = saturate(x_quan, self.activation_wordlength)
            x_quan = linear_dequantize(x_quan, self.scaling_factor, self.zero_point)
            
            return x_quan


def activation_quantization(model, wordlength, quantization_method, calibrate_loader):
    # add activation quantisation module
    replace_dict ={}
    for name, module in model.named_modules(): 
        if type(module) in [nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear]:
            module_quan = nn.Sequential(*[QuanAct(wordlength, quantization_method), copy.deepcopy(module), QuanAct(wordlength, quantization_method)]) 
            replace_dict[module] = module_quan

    for name, module in model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                submodule_quan = replace_dict[submodule] 
                assert(hasattr(module, subname))
                setattr(module,subname,submodule_quan)

    model.eval() 
    if torch.cuda.is_available():
        model = model.cuda()

    # gather activation data
    with torch.no_grad():
        for i, (images, target) in enumerate(calibrate_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            model(images)

    for name, module in model.named_modules():
        if isinstance(module, QuanAct):
            module.gather_data = False

    # find scaling factor and zero point
    bFirst = True
    for module in model.modules():
        if isinstance(module, QuanAct):
            if bFirst:
                bFirst = False
                network_min = module.x_min
                network_max = module.x_max
            else:
                network_min = torch.minimum(network_min, module.x_min)
                network_max = torch.maximum(network_max, module.x_max)

    print("activation min:", network_min)
    print("activation max:", network_max)


    for module in model.modules():
        if isinstance(module, QuanAct):
            if quantization_method in [1, 2]:
                module.x_min = network_min
                module.x_max = network_max
                module.get_scale_shift()
            elif quantization_method in [3, 4]:
                module.get_scale_shift()
            else:
                assert False

    return model
