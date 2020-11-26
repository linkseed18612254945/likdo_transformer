import torch


def get_gpu_device(use_gpu=True, gpu_index=0):
    device = torch.device(f'cuda:{gpu_index}') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
    return device