import torch
from models import create_classifier, create_dense_predictor, create_ssl
import utils


in_channels = 3
in_size = (32, 32)
n_classes = 10

model = create_classifier(
    enc_from_ssl=False,
    enc_config_dir="sft",
    enc_config="c3.yaml",
    head_config_dir="linear",
    head_config="c1.yaml",
    in_channels=in_channels,
    in_size=in_size,
    n_classes=n_classes,
)

x = torch.randn(1, in_channels, *in_size)
pred = model(x)
loss = torch.mean(pred)
loss.backward()

unused = [k for k, p in model.named_parameters() if p.grad is None]
used = [k for k, p in model.named_parameters() if p.grad is not None]
n_unused = len(unused)
n_used = len(used)
total = n_unused + n_used

if n_unused:
    head = f"Unused params ({n_unused}/{total})"
    print(utils.join_head_body(head, '\n'.join(unused)))
else:
    print("No unused params")