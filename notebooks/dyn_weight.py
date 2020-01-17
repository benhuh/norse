import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init


class Active_Weight(nn.Module):
    def __init__(self, n_input, n_output, n_context, gain_active=1, gain_passive=1, passive=True):  # n_bottleneck
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.w_passive, self.b_passive = 0, 0

        if passive:
            w, b = self.initialize(n_input, n_output, gain_passive)
            self.w_passive = Parameter(w.view(-1))
            self.b_passive = Parameter(b)

        if n_context > 0:
            w_all, b_all = [], []
            for i in range(n_context):
                w, b = self.initialize(n_input, n_output, gain_active)
                w_all.append(w)
                b_all.append(b)
            self.w_active = Parameter(torch.stack(w_all, dim=2).view(-1, n_context))
            self.b_active = Parameter(torch.stack(b_all, dim=1))

    def initialize(self, in_features, out_features, gain):
        weight = torch.Tensor(out_features, in_features)
        bias = torch.Tensor(out_features)
        self.reset_parameters(weight, bias)
        return gain * weight, gain * bias
    
    def reset_parameters(self, weight, bias=None):
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)
            
    def forward(self, context):
        n_batch = context.shape[0]
        weight = F.linear(context, self.w_active, self.w_passive).view(n_batch, self.n_output, self.n_input)
        bias = F.linear(context, self.b_active, self.b_passive).view(n_batch, self.n_output)
        return weight, bias


class Linear_Active(nn.Module):
    def __init__(self, n_input, n_output, n_context, gain_w, gain_b, passive): 
        super().__init__()
        self.active_weight = Active_Weight(n_input, n_output, n_context, gain_w, gain_b, passive)
        
    def forward(self, x, context):
        weight, bias = self.active_weight(context)
        x = torch.einsum('bs,bas->ba', x, weight) + bias
        return x


class Model_Active(nn.Module):
    def __init__(self, n_arch, n_context, gain_w, gain_b, nonlin=nn.ReLU(), passive=True): 
        super().__init__()
        self.nonlin = nonlin
        self.depth = len(n_arch) - 1
        
        module_list = []
        for i in range(self.depth):
            module_list.append(
                Linear_Active(n_arch[i], n_arch[i + 1], n_context, gain_w, gain_b, passive))

        self.module_list = nn.ModuleList(module_list)
        
    def forward(self, x, context):
        for i, module in enumerate(self.module_list):
            x = module(x, context)
            if i < self.depth - 1:
                x = self.nonlin(x)
        return x


class Model_Onehot(nn.Module):
    def __init__(self, model, n_task, n_context): 
        super().__init__()
        self.model = model
        self.onehot2context = nn.Linear(n_task, n_context, bias=False)
        
        with torch.no_grad():
            self.onehot2context.weight.zero_()  # Initialize to zero
        
    def forward(self, input, onehot):
        context = self.onehot2context(onehot)
        out = self.model(input, context)
        return out

    def predict_with_context(self, input, context):
        out = self.model(input, context)
        return out


class Model_Test(nn.Module):
    def __init__(self, model, n_task, n_context): 
        super().__init__()
        self.model = model
        self.n_task = n_task
        self.n_context = n_context
        self.context = Parameter(torch.zeros([n_task, n_context]))
        
        if n_task > 1:
            raise NotImplementedError("Should be fixed")
        
    def forward(self, input, onehot=None):
        n_batch = input.shape[0]
        context = self.context.repeat(n_batch, 1)
        out = self.model(input, context)
        return out

    def reset_context(self):
        self.context = Parameter(torch.zeros([self.n_task, self.n_context]))


# class Model_Onehot_Orig(nn.Module):
#     def __init__(self, model, onehot2context): 
#         super().__init__()
#         self.model = model
#         self.onehot2context = onehot2context

#     def forward(self, input, onehot):
#         context = self.onehot2context(onehot)
#         out = self.model(input, context)
#         return out


# class Onehot2Context(nn.Module):
#     def __init__(self, n_task, n_context, gain=0): 
#         super().__init__()
#         raise ValueError("DK: Not used, so can be removed?")
#         self.weight = nn.Parameter(gain * torch.randn(n_context, n_task))

#     def forward(self, onehot):
#         return onehot @ self.weight.t()


# class Onehot2Context_New(nn.Module):
#     def __init__(self, n_task, n_context, gain=0): 
#         super().__init__()
#         raise ValueError("DK: Not used, so can be removed?")
#         self.linear = nn.Linear(n_task, n_context, bias=False)
#         with torch.no_grad():
#             self.linear.weight.zero_()  # Initialize to zero
        
#     def forward(self, onehot):
#         return self.linear(onehot)
