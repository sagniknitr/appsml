import torch
import torch.fx
import librespeech
import pytorch_SUT
import perftools
import numpy as np

rnnt = pytorch_SUT.PytorchSUT(
    './pytorch/configs/rnnt.toml', '/research/data/mlmodels/rnnt.pt')

model = rnnt.model

print(model)


for n, p in model.named_parameters(): print(n, p.shape)
for n, p in model.named_buffers(): print(n, p.shape)

params = {
    name: p.detach().numpy()
    for name, p in model.named_parameters()
}

params.update({
    name: b.detach().numpy()
    for name, b in model.named_buffers()
})

# np.savez('rnnt.npz', **params)
