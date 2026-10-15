import torch

# Check if the Intel GPU is detected
print(f"XPU Available: {torch.xpu.is_available()}")

# Move a tensor to the 12 Xe computing cores
tensor = torch.tensor([1.0, 2.0]).to("xpu")
print(tensor)