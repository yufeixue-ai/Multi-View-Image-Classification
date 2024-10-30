import torch
import torch.nn as nn

from loguru import logger

def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class BaseQuantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError

class FeatQuantizer(BaseQuantizer):
    def __init__(self, bit, all_positive=True, symmetric=False):
        super().__init__(bit)

        assert all_positive == True, "FeatQuantizer only supports all_positive=True"
        assert symmetric == False, "FeatQuantizer only supports symmetric=False"

        if bit == 1:
            self.thd_pos = 1
            self.thd_neg = 0
        else:
            self.thd_neg = -2 ** (bit-1)
            self.thd_pos = 2 ** (bit-1) - 1
        

        self.s = torch.nn.Parameter(torch.ones(1))
        
        self.is_init = False

    def init_from(self, x, *args, **kwargs):
        self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        # logger.debug(f'self.is_init: {self.is_init}')
        if not self.is_init:
            self.init_from(x)
            self.is_init = True
            
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)
        
        # logger.debug(f's: {self.s.item()}')
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x