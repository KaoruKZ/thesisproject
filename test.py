import torch
print(torch.cuda.is_available())  # Should be True
x = torch.rand(10, 10).cuda()
y = torch.rand(10, 10).cuda()
print(x + y) # Should print a tensor