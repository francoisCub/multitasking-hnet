from torch import Tensor, LongTensor, mean, median, min, nn, ones, sum, no_grad, exp, randn
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector


def entropy(input):
    log_p = nn.functional.log_softmax(input, dim=1)
    p = exp(log_p)
    return -sum(p*log_p, dim=1).mean()

def compute_nbr_params(model):
    total_size = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.is_sparse:
            total_size += p.coalesce().values().numel()
        else:
            total_size += p.numel()
    return total_size


def estimate_connectivity(hnet, latent_size):
    # when target model is huge and latent size is not small, does not scale
    # now a memory friendlier version
    with no_grad():
        z = nn.functional.one_hot(LongTensor(
            range(latent_size)), latent_size).float().cuda()
        res = []
        for vector_z in z:
            out = hnet(vector_z.unsqueeze(0))
            res.append(sum(out != 0, dim=1) / out.shape[1])
        res = Tensor(res)
        z = ones(1, latent_size).cuda()
        out = hnet(z)
        coverage = sum(out != 0, dim=1) / out.shape[1]
        return mean(res), min(res), max(res), median(res), res, mean(coverage)

def estimate_target_sparsity(model, latent_size, type, layers_type=None, device="cuda"):
    with no_grad():
        if type == "hnet":
            z = randn(1, latent_size, device=device)
            out = model.forward_hnet(z)
            num_zeros = 0
            num_elements = 0
            for name, param in out.items():
                num_zeros += sum(param==0).item()
                num_elements += param.nelement()
            return num_zeros / num_elements
        elif type == "experts":
            param_vector = parameters_to_vector(model.parameters())
            num_zeros = 0
            num_elements = 0

            for module in model.modules():
                for buffer_name, buffer in module.named_buffers():
                    if "weight_mask" in buffer_name:
                        num_zeros += sum(buffer == 0).item()
                        num_elements += buffer.nelement()
                    if "bias_mask" in buffer_name:
                        num_zeros += sum(buffer == 0).item()
                        num_elements += buffer.nelement()
            return num_zeros / num_elements
        else:
            raise ValueError("hnet or experts")
    

# delete_param and set_param from https://discuss.pytorch.org/t/how-does-one-have-the-parameters-of-a-model-not-be-leafs/70076/10


def delete_param(module: nn.Module, module_list):
    if len(module_list) == 1:
        delattr(module, module_list[0])
    else:
        delete_param(getattr(module, module_list[0]), module_list[1:])


def set_param(module: nn.Module, module_list, param):
    if len(module_list) == 1:
        setattr(module, module_list[0], param)
    else:
        set_param(getattr(module, module_list[0]), module_list[1:], param)

def get_param(module: nn.Module, module_list):
    if len(module_list) == 1:
        return getattr(module, module_list[0])
    else:
        return get_param(getattr(module, module_list[0]), module_list[1:])

def sparsify_resnet(model, sparsity):
    all_param = []
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            for name, param in module.named_parameters():
                all_param.append((module, name.split('.')[-1]))
    prune.global_unstructured(
    all_param,
    pruning_method=prune.RandomUnstructured,
    amount=sparsity,
)   

    return model