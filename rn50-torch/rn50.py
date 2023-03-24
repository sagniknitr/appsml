import torch
import torchvision

dev = torch.device('cuda:0')

# torch.set_num_interop_threads(1)
# torch.set_num_threads(1)

net = torchvision.models.resnet50(pretrained=False).half().to(dev)
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
x = torch.randn(1, 3, 224, 224, device=dev, dtype=torch.half)

# start of region of interest
opt.zero_grad()
yp = net(x) # fwd pass
loss = torch.sum(yp)
loss.backward() # backprop
opt.step()



