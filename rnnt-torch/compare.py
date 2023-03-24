import torch
import torch.fx
import librespeech
import pytorch_SUT
import perftools
import rnnt
import rnnt_weights


dataset = librespeech.Librespeech(
    '/research/data/mldata/LibriSpeech/librispeech-dev-clean-wav.json')

mlperf_rnnt = pytorch_SUT.PytorchSUT(
    './pytorch/configs/rnnt.toml', '/research/data/mlmodels/rnnt.pt')


new_rnnt = rnnt.Rnnt()
new_rnnt.eval()
new_rnnt.load_from_file('/research/data/mlmodels/npz/rnnt.npz')


x = torch.Tensor(dataset[0].audio.samples).unsqueeze_(0)

def run_mlperf(x):
    l = torch.LongTensor([dataset[0].audio.num_samples])
    x, l = mlperf_rnnt.audio_preprocessor.forward((x, l))
    x = x.permute(2, 0, 1)
    logits, logits_lens, output = mlperf_rnnt.greedy_decoder.forward(x, l)
    return output

def run_new(x):
    l = torch.LongTensor([dataset[0].audio.num_samples])
    print(x.shape, l)
    return new_rnnt(x, l)


y_mlperf = run_mlperf(x.clone())
y_new = run_new(x.clone())
print(y_mlperf)
print(y_new)
