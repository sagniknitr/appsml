import torch
import torch.fx
from torch.fx.node import Node, map_aggregate

class MyCustomTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        print(f'is_leaf_module({module_qualified_name})')
        return False
        return m.__module__.startswith('torch.nn')

# class PerfRecorder(torch.fx.Interpreter):
#     def call_function(self, target, args, kwargs):
#         opname = str(target.__name__)
#         print(f'call_function: {opname}')
#         return super().call_function(target, args, kwargs)

#     def call_method(self, target, args, kwargs):
#         opname = str(target)
#         print(f'call_method: {opname}')
#         return super().call_method(target, args, kwargs)

#     def call_module(self, target, args, kwargs):
#         opname = type(self.fetch_attr(target)).__name__
#         out = super().call_module(target, args, kwargs)
#         print(f'call_module: {opname}')

#         return out

class Test(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(*[
            torch.nn.Linear(3, 3),
        ])

    def forward(self, x):
        print(f'x: {type(x)}')
        y = self.mlp(x)

        print(f'y: {type(y)}')
        return y


t = Test()

gm = MyCustomTracer().trace(t, concrete_args=dict(
    x=torch.zeros((3,)),
    y=torch.zeros((3,))
))

# PerfRecorder(gm).run(torch.rand((3,)))

