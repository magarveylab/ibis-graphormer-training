import torch
from torch import nn

def compile_with_torchscript(model: nn.Module, model_fp: str):
    model.eval()
    if callable(getattr(model, "jittable", None)):
        model.jittable()
    compiled_model = torch.jit.script(model)
    torch.jit.save(compiled_model, model_fp)