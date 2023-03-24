import torch
import torch.fx
import librespeech
import pytorch_SUT
import perftools

class DeepTracer(torch.fx.Tracer):
    # def is_leaf_module(self, _, __): return False

    def call_module(self, m: torch.nn.Module, forward, args, kwargs):
        module_qualified_name = self.path_of_module(m)
        # if not self.is_leaf_module(m, module_qualified_name):
        return forward(*args, **kwargs)
        # return self.create_proxy('call_module', module_qualified_name, args, kwargs)

    # def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
    #     return attr_val

    @staticmethod
    def symbolic_trace(root : torch.nn.Module, concrete_args = None):
        tracer = DeepTracer()
        graph = tracer.trace(root, concrete_args)
        name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
        return torch.fx.GraphModule(tracer.root, graph, name)


class PerfRecorder(torch.fx.Interpreter):
    def run_node(self, n : torch.fx.Node):
        print(n)
        return super().run_node(n)

    def call_function(self, target, args, kwargs):
        opname = str(target.__name__)
        if opname == 'matmul':
            print(f'Matmul({args[0].shape} * {args[1].shape})')
        else:
            print(opname)
        with perftools.pinperf.perf_roi(1, str(target), opname):
            return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        opname = str(target)
        print(opname)
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

            print(f'ops.Conv2D(dtype, n, {h}, {w}, {c}, {p}, {q}, {k}, {r}, {s}, {stride}),')

        else:
            print(opname)

        return out

dataset = librespeech.Librespeech(
    '/research/data/mldata/LibriSpeech/librispeech-dev-clean-wav.json')

rnnt = pytorch_SUT.PytorchSUT(
    './pytorch/configs/rnnt.toml', '/research/data/mlmodels/rnnt.pt')

print(rnnt)

print(dataset[0])

x = torch.Tensor(dataset[0].audio.samples).unsqueeze_(0)
print(x.shape)
l = torch.LongTensor([dataset[0].audio.num_samples])



with perftools.pinperf.perf_roi(0, 'rnnt', 'rnnt'):
    x, l = rnnt.audio_preprocessor.forward((x, l))
    x = x.permute(2, 0, 1)
    print('x.shape = ', x.shape)
    logits, logits_lens, output = rnnt.greedy_decoder.forward(x, l)

print(logits.shape)

# logits, logits_lens = self._model.encoder(x, out_lens)

# gm = torch.fx.symbolic_trace(
#     rnnt.greedy_decoder._model.encoder, dict(x=x, x_padded=x, x_lens=l, out_lens=l))

# with perftools.pinperf.perf_roi(0, 'rnnt', 'rnnt'):
#     PerfRecorder(gm).run(x, l)
