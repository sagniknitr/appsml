import torch
import torch.jit
import torch.fx
import bert
import bert_weights
import numpy as np
import perftools

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

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

        if opname == 'Linear':
            m = self.fetch_attr(target)
            x = args[0]

            print(f'Linear({x.shape} * {m.weight.shape} + {m.bias.shape})')

        else:
            print(opname)
        with perftools.pinperf.perf_roi(1, target, opname):
            return super().call_module(target, args, kwargs)


cfg = bert.bert_large_conf(512)
bert = bert.BertSquad(cfg)
# model.load_from_file('./bert-large-squad.npz')


# input_ids = torch.randint(0, cfg.vocab_size, (N, 512)).to(device)
# token_type_ids = torch.randint(0, 1, (N, 512)).to(device)
# opt = torch.optim.SGD(bert.parameters(), lr=0.001, momentum=0.9)

# with torch.autograd.profiler.emit_nvtx():
#     t0 = time.perf_counter_ns()
#     for i in range(I):
#         print(f'Step {i} / {I}')
#         opt.zero_grad()
#         yp = bert(input_ids, token_type_ids)
#         loss = torch.sum(yp)
#         loss.backward()
#         opt.step()
#     t1 = time.perf_counter_ns()

input_ids = torch.randint(0, cfg.vocab_size, (1, 254))
token_type_ids = torch.randint(0, 1, (1, 254))
opt = torch.optim.SGD(bert.parameters(), lr=0.001, momentum=0.9)

with perftools.pinperf.perf_roi(3, 'infer', 'infer'):
    # yp = bert(input_ids, token_type_ids)
    opt.zero_grad()
    yp = bert(input_ids, token_type_ids)
    loss = torch.sum(yp)
    loss.backward()
    opt.step()

# tokens = model.tokenizer(question, text, return_tensors='pt')

# gm = torch.fx.symbolic_trace(bert, concrete_args=dict(
#     input_ids=input_ids,
#     seg_ids=token_type_ids
# ))

# with perftools.pinperf.perf_roi(0, 'bert', 'bert'):
#     PerfRecorder(gm).run(input_ids, token_type_ids)
