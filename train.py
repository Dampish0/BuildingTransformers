import os
import requests
import torch
import numpy as np
import torch.optim.adamw
from model import Model
torch.set_float32_matmul_precision('high')
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

print(len(data))

from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")

vocab_size = tokenizer.vocab_size
tokens = tokenizer.encode(data)

# ##data = "data"
# vocab = sorted(list(set(data)))
# vocab_size = len(vocab)

# print(vocab.index('t'))

# tokens = [vocab.index(v) for i, v in enumerate(data)]

# print(tokens[:40])
# #print(vocab)

maxSteps = 3000
gradAccum = 1
batchSize = 16
n_head = 8
n_layers = 20 #32
n_embd = 256 #512
blocksize = 1024

#MHA
model = Model(n_layers,n_embd,n_head,vocab_size,blocksize).to("cuda")

#model = Model(n_layers,n_embd,n_head,vocab_size,blocksize, LCompression=256, flashATTN=True, attn_dropout=0.4).to("cuda")
model = torch.compile(model)
tot = 0
for param in model.parameters():
    tot += param.numel()

print(f"total params: {tot//1e6}M params")


optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)
def get_batch():
    ix = torch.randint(len(tokens) - blocksize, (batchSize,))
    x = torch.stack([torch.tensor(tokens[i:i+blocksize], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(tokens[i+1:i+blocksize+1], dtype=torch.long) for i in ix])
    return x.to("cuda"), y.to("cuda")

import time
for i in range(maxSteps):
    xo = time.time()
    optimizer.zero_grad()
    for _ in range(gradAccum):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x, y = get_batch()
            out, loss = model(x, y)
        loss.backward()
    optimizer.step()
    if i % 10 == 0:
        t1 = time.time() - xo
        print(f"step: {i}/{maxSteps}, loss: {loss.item():.6f}, t/step: {t1:.4f}s, time left: {t1 * (maxSteps-i):.2f}")


generated = torch.tensor([0], device="cuda").unsqueeze(0)
from torch import nn
t0 = time.time()
for i in range(120):
    out = model(generated)
    out = out[:,-1,:]
    
    out = nn.functional.softmax(out, dim=-1)
    predchar = torch.multinomial(out, num_samples=1)
    
    
    generated = torch.concat([generated, predchar], dim=-1)

#generated = model.generate(torch.tensor([0], device="cuda").unsqueeze(0), 200)
t1 = time.time() - t0
print(f"time to generate: {t1}, tokens per second: {200/t1}")
#outText = "".join([vocab[int(v.item())] for i, v in enumerate(generated.squeeze(0))])

outText = tokenizer.decode(generated.squeeze(0))
print(outText)