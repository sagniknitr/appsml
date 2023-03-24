import torch
import librespeech
import pytorch_SUT
import pyaudio
import wave
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 44100
RECORD_SECONDS = 10

audio = pyaudio.PyAudio()

print("----------------------record device list---------------------")
for ii in range(audio.get_device_count()):
    print(audio.get_device_info_by_index(ii))
print("-------------------------------------------------------------")

stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=6,
    frames_per_buffer=CHUNK)

print ("recording started")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    print(data.shape)
    frames.append(data)

stream.stop_stream()
stream.close()
audio.terminate()

raw_audio = np.concatenate(frames)
seg = librespeech.AudioSegment(raw_audio, 44100, 16000)

rnnt = pytorch_SUT.PytorchSUT(
    './pytorch/configs/rnnt.toml', '/research/data/mlmodels/rnnt.pt')

print(rnnt)


x = torch.Tensor(seg.samples).unsqueeze_(0)
print(x.shape)
l = torch.LongTensor([seg.num_samples])

x, l = rnnt.audio_preprocessor.forward((x, l))

print('x.shape = ', x.shape)

_, _, transcript = rnnt.greedy_decoder.forward(x.permute(2, 0, 1), l)

print()
print()
print(''.join([rnnt.labels[i] for i in transcript[0]]))
