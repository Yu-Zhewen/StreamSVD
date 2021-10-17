import sys

from hybrid_svd.common.utils import * 

from heapq import nsmallest
from operator import itemgetter

def pruning_target_module_filter(module_instance, interested_type=[nn.Conv2d]):
    for module_type in interested_type:
        if isinstance(module_instance, module_type):
            return True
    return False            

class PrunningFineTuner:
    def __init__(self, train_data_loader, test_data_loader, model_name, model, pruning_ratio, random_input):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.model_name = model_name
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.random_input = random_input

        if torch.cuda.is_available():
            self.loss_criterion = nn.CrossEntropyLoss().cuda()
        else:
            self.loss_criterion = nn.CrossEntropyLoss()


    # https://github.com/NVlabs/Taylor_pruning/blob/master/pruning_engine.py
    def get_taylor_candidates(self, num_filters_to_prune, target_type=[nn.BatchNorm2d]):

        handler_collection = []
        
        for name, module in self.model.named_modules():
            if pruning_target_module_filter(module, target_type):
                def just_hook(self, grad_input, grad_output):
                    self.output_grad = (grad_output[0]*self.output).sum(-1).sum(-1).mean(0)

                def forward_hook(self, input, output):
                    self.output = output
                
                handler_collection.append(module.register_forward_hook(forward_hook))
                handler_collection.append(module.register_backward_hook(just_hook))
    

        self.model.eval()

        for i, (images, target) in enumerate(self.test_data_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = self.model(images)
            loss = self.loss_criterion(output, target)

            self.model.zero_grad()
            loss.backward()

            for name, module in self.model.named_modules():
                if pruning_target_module_filter(module, target_type):
                    nunits = module.output_grad.size(0)
                    criteria_for_layer = (module.output_grad).data.pow(2)

                    if i == 0:
                        module.prune_network_accomulate = criteria_for_layer
                    else:
                        module.prune_network_accomulate += criteria_for_layer

        globalRanking = []
        conv_index = 0
        for name, module in self.model.named_modules():
            if pruning_target_module_filter(module, target_type):
                if num_filters_to_prune == -1 or (self.model_name == "resnet18" and num_filters_to_prune != -1 and (".conv1" in name or ".bn1" in name)):
                    metric = module.prune_network_accomulate.cpu().numpy()
                    globalRanking += [(conv_index, i, x) for i,x in enumerate(metric)]
                    delattr(module, "prune_network_accomulate")
                    delattr(module, "output")
                    delattr(module, "output_grad")
                    conv_index += 1
                
        for handler in handler_collection:
            handler.remove()

        if num_filters_to_prune == -1:
            return globalRanking           
        else:
            return nsmallest(num_filters_to_prune, globalRanking, itemgetter(2))   
