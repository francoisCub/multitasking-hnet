import copy

from torch import cat, jit, nn, zeros_like

from hypernetworks.utils import delete_param, set_param


class TargetModel(nn.Module):
    def __init__(self, target_model: nn.Module, mode="one_block") -> None:
        super().__init__()
        self.target_model = copy.deepcopy(target_model)
        named_parameters = list(self.target_model.named_parameters())
        self.params_info = [(name, param.shape, param.numel())
                            for name, param in named_parameters if param.requires_grad == True]
        for (name, param) in named_parameters:
            delete_param(self.target_model, name.split("."))
            set_param(self.target_model, name.split("."), zeros_like(param))

    def forward(self, x, params):
        for (name, param) in params.items():
            set_param(self.target_model, name.split("."), param)
        return self.target_model(x)

    def get_params_info(self):
        return iter(self.params_info)

    def get_params_names(self):
        return iter([name for (name, _, _) in self.params_info])


class BatchTargetModel(nn.Module):
    def __init__(self, batch_size, target_model) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("Invalid batch size")
        self.batch_size = batch_size
        self.models = nn.ModuleList(
            [TargetModel(copy.deepcopy(target_model)) for _ in range(batch_size)])

    def forward(self, x, params):
        futures = [jit.fork(model, x.unsqueeze(0), p)
                   for (model, x, p) in zip(self.models, x, params)]
        results = [jit.wait(fut) for fut in futures]
        return cat(results, dim=0)
        # return cat(results, dim=0)

    def get_params_info(self):
        return self.models[0].get_params_info()
