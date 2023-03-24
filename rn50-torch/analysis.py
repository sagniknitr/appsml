import torch
import torch.fx
import torchvision
import perftools

from torch.fx.node import Node, map_aggregate


class PerfRecorder(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        opname = str(target.__name__)
        if opname == 'matmul':
            print(f'Matmul({args[0].shape} * {args[1].shape})')
        with perftools.pinperf.perf_roi(1, str(target), opname):
            return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        opname = str(target)
        with perftools.pinperf.perf_roi(1, target, opname):
            return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        opname = type(self.fetch_attr(target)).__name__


        with perftools.pinperf.perf_roi(1, target, opname):
            out = super().call_module(target, args, kwargs)

        if opname == 'Linear':
            m = self.fetch_attr(target)
            x = args[0]

            print(f'Linear({x.shape} * {m.weight.shape} + {m.bias.shape})')

        elif opname == 'Conv2d':
            m = self.fetch_attr(target)
            x = args[0]

            [n, c, h, w] = x.shape
            [_, k, p, q] = out.shape
            r, s = m.kernel_size
            stride = m.stride[0]
            padding = m.padding[0]
            # print(f'Padding: {padding}')

            print(f'ops.Conv2D(dtype, {n}, {h}, {w}, {c}, {p}, {q}, {k}, {r}, {s}, {stride}, {padding}, False),')

        else: ...

        return out

net = torchvision.models.resnet50(pretrained=True)

gm = torch.fx.symbolic_trace(net)

torch.nn.Conv2d

with perftools.pinperf.perf_roi(0, 'rn50', 'rn50'):
    PerfRecorder(gm).run(torch.rand(1, 3, 224, 224))

