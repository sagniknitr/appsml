import torch
import torch.fx
import librespeech
import pytorch_SUT
import perftools
import numpy as np


dev_clean = librespeech.Librespeech(
    '/research/data/mldata/LibriSpeech/librispeech-dev-clean-wav.json')

train_clean_100 = librespeech.Librespeech(
    '/research/data/mldata/LibriSpeech/librispeech-train-clean-100-wav.json')
train_clean_360 = librespeech.Librespeech(
    '/research/data/mldata/LibriSpeech/librispeech-train-clean-360-wav.json')
train_other_500 = librespeech.Librespeech(
    '/research/data/mldata/LibriSpeech/librispeech-train-other-500-wav.json')



rnnt = pytorch_SUT.PytorchSUT(
    './pytorch/configs/rnnt.toml', '/research/data/mlmodels/rnnt.pt')


def count_dataset(data):
    il = 0
    ol = 0
    ns = len(data)

    for i in range(len(data)):
        print(f'{i+1}/{ns}')
        l = torch.LongTensor([data[i].audio.num_samples])
        x = torch.Tensor(data[i].audio.samples).unsqueeze_(0)

        x, l = rnnt.audio_preprocessor.forward((x, l))
        x = x.permute(2, 0, 1)

        _, logits_lens, _ = rnnt.greedy_decoder.forward(x, l)

        il += l.item()
        ol += logits_lens.item()


    return np.array([il, ol, ns])


stats = np.array([0, 0, 0])
stats += count_dataset(train_clean_100)
stats += count_dataset(train_clean_360)
stats += count_dataset(train_other_500)

il, ol, ns = stats[0], stats[1], stats[2]
print(il, ol, ns)
print(il/ns, ol/ns)

