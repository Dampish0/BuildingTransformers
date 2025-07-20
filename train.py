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

maxSteps = 2
gradAccum = 4
batchSize = 4
n_head = 8
n_layers = 8
n_embd = 256
blocksize = 256


model = Model(n_layers,n_embd,n_head,vocab_size).to("cuda")

tot = 0
for param in model.parameters():
    tot += param.numel()

print(f"total params: {tot//1e6}M params")


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)



for i in range(maxSteps):
    optimizer.zero_grad()
    for i in range(gradAccum):
        x = torch.stack([torch.tensor(tokens[d * blocksize: (d+1) * blocksize], dtype=torch.long) for d in range(batchSize)]).to("cuda")
        y = torch.stack([torch.tensor(tokens[(d+1) * blocksize:(d+2) * blocksize], dtype=torch.long) for d in range(batchSize)]).to("cuda")
        out, loss = model(x, y)
        loss.backward()
    optimizer.step()
    print(loss.item())


generated = torch.tensor([0], device="cuda").unsqueeze(0)
from torch import nn
for i in range(100):
    out = model(generated)
    out = out[:,-1,:]
    
    out = nn.functional.softmax(out, dim=-1)
    predchar = torch.multinomial(out, num_samples=1)
    
    
    generated = torch.concat([generated, predchar], dim=-1)


outText = [vocab[int(v.item())] for i, v in enumerate(generated.squeeze(0))]
print("".join(outText))