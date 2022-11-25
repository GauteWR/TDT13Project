from pynvml import *
import torch

def print_gpu_usage():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_usage()
torch.ones((1, 1)).to('cuda')
print_gpu_usage()