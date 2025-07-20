import os
import requests
import torch
import numpy as np
import torch.optim.adamw
from model import Model

input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

print(len(data))
##data = "data"
vocab = sorted(list(set(data)))
vocab_size = len(vocab)

print(vocab.index('t'))

tokens = [vocab.index(v) for i, v in enumerate(data)]

print(tokens[:40])
print(vocab)

n_head = 8
n_layers = 8
n_embd = 256
blocksize = 256
from model import Model

model = Model(n_layers,n_embd,n_head,vocab_size).to("cuda")

tot = 0
for param in model.parameters():
    tot += param.numel()

print(f"total params: {tot//1e6}M params")


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

#orch.stack()

for i in range(100):
    optimizer.zero_grad()
    x = torch.tensor(tokens[i * blocksize: (i+1) * blocksize], dtype=torch.long).to("cuda")
    y = torch.tensor(tokens[(i+1) * blocksize:(i+2) * blocksize], dtype=torch.long).to("cuda")
    out, loss = model(x.unsqueeze(0), y.unsqueeze(0))
    loss.backward()
    optimizer.step()
    print(loss.item())