import torch
import transformers.models.bert.configuration_bert as CB
import transformers.models.bert.modeling_bert as MB


config = CB.BertConfig()
l = MB.BertLayer(config)

traced = torch.jit.trace(l, torch.rand(1, 512, 768))
traced.op
print(traced.inlined_graph)