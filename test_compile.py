import torch
import torch.backends.cuda
import numpy as np
from models.encoders.sft.sft import *


def init_model():
    model = Transformer(16, False, 768, 3072, 12, 0.1, None, True)
    return model.cuda()

def generate_data():
    x = torch.randn(8, 180, 768)
    ctx = torch.randn(8, 160, 768)
    return x.cuda(), ctx.cuda()

def evaluate(model, x):
    with torch.no_grad():
        return model(x[0], x[1])


def timed(func):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000


torch.backends.cuda.matmul.allow_tf32 = True
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

n_iters = 10

model = init_model()
opt_model = torch.compile(model, fullgraph=True, backend="inductor")

print("~" * 10)

eager_times = []
for i in range(n_iters):
    x = generate_data()
    time = timed(lambda: evaluate(model, x))
    eager_times.append(time)
    print(f"eager time {i}: {time:.3g}")
print("~" * 10)

compile_times = []
for i in range(n_iters):
    x = generate_data()
    time = timed(lambda: evaluate(opt_model, x))
    compile_times.append(time)
    print(f"compile time {i}: {time:.3g}")
print("~" * 10)

eager_median = np.median(eager_times)
compile_median = np.median(compile_times)
speedup = eager_median / compile_median
print("\n".join([
    f"eager median: {eager_median:.3g}",
    f"compile median: {compile_median:.3g}",
    f"compile time: {compile_times[0]:.3g}",
    f"speedup: {speedup:.3g}",
]))
print("~" * 10)