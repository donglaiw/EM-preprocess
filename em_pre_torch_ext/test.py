import torch
import em_pre_cuda

a = torch.rand((100, 100, 7), device="cuda")
b = torch.tensor([0, 0, 3], device="cuda", dtype=torch.float32)
c = em_pre_cuda.median_filter(a,b)
print(c, c.size())
