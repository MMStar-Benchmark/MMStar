import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .llava_next import LLaVA_Next
