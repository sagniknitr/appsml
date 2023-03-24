import torch
import torch.jit
import torch.fx
import bert
import bert_weights
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class PerfRecorder(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        opname = str(target.__name__)
        print(opname)
        return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        opname = str(target)
        print(opname, [f'Tensor[{a.shape}]' if isinstance(a, torch.Tensor) else a for a in args])
        return super().call_method(target, args, kwargs)

    def call_module(self, target, args, kwargs):
        opname = type(self.fetch_attr(target)).__name__
        print(opname)
        return super().call_module(target, args, kwargs)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")

theirs = AutoModelForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")


ours = bert.Bert(bert.bert_large_conf(512))
ours.load_weights('./bert-large-squad.npz')

x = torch.randn(2, 512, 1024).detach()



theirs0 = theirs.bert.encoder
ours0 = ours

# print('Parameters in THEIRS =')
# for n, p in theirs0.named_parameters(): print(n, p.shape)
# print('Parameters in OURS =')
# for n, p in ours0.named_parameters(): print(n, p.shape)
# print()



# gm = torch.fx.symbolic_trace(theirs0, concrete_args=dict(
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_value=None,
#         output_attentions=False))

# PerfRecorder(gm).run(x)

# gm = torch.fx.symbolic_trace(ours0)
# PerfRecorder(gm).run(x)

o1 = theirs0.forward(x).last_hidden_state
o2 = ours0(x)

print('o1 = ', o1.shape)
print('o2 = ', o2.shape)
print(torch.isclose(o1, o2).all())