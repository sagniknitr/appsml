import torch
import torchvision
import torch_tensorrt
import torch.backends.cudnn as cudnn
import numpy as np
import time

cudnn.benchmark = True

def benchmark(model, input_shape=(1024, 3, 512, 512), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    if dtype=='fp16': input_data = input_data.to(torch.float16)
    input_data = input_data.to("cuda")


    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print('Average throughput: %.2f images/second'%(input_shape[0]/np.mean(timings)))

# device = torch.device("cuda:0")

net = torchvision.models.resnet50(pretrained=False).eval().to("cuda")

net = torch.jit.trace(net, torch.randn(1, 3, 224, 224).to("cuda"))
net.eval().to("cuda")

print(type(net))

imagenet_data = torchvision.datasets.ImageNet('/imagenet', 'val')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=1,
                                          shuffle=True)

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(data_loader,
                                              cache_file='./calibration.cache',
                                              use_cache=False,
                                              algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                                              device=torch.device('cuda:0'))

spec = {
    'forward': torch_tensorrt.ts.TensorRTCompileSpec(**{
            "inputs": [ torch.randn(1, 3, 224, 224).to("cuda")],
            "enabled_precisions": {torch.float, torch.half, torch.int8},
            "refit": False,
            "debug": False,
            "device": {
                "device_type": torch_tensorrt.DeviceType.GPU,
                "gpu_id": 0,
                "dla_core": 0,
                "allow_gpu_fallback": True
            },
            "capability": torch_tensorrt.EngineCapability.default,
            "num_min_timing_iters": 2,
            "num_avg_timing_iters": 1,
            "calibrator": calibrator,
    })
}

print('Compiling')
# trt_model = torch_tensorrt.compile(net,
#     inputs= [torch_tensorrt.Input((1, 3, 224, 224), dtype=torch.float)],
#     enabled_precisions= { torch.int8 },
# )

trt_model = torch._C._jit_to_backend("tensorrt", net, spec)
# trt_model = torch_tensorrt.compile(net, **compile_spec)

benchmark(trt_model, input_shape=(1, 3, 224, 224), nruns=100, dtype="fp32")

