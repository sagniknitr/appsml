import torch
import bert

bert_med = bert.Bert(bert.BertConfig())

x = torch.randn(2, 512, 768)


y = bert_med(x)
print(y.shape)
