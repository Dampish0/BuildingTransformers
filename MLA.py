import torch
from torch import nn
import math
import os
torch.manual_seed(42)

# changed into a parameter into the model for easier use
#LCompression = 576//2
    
import torch.nn.functional as F

class CausalSelfAttentionMLA(nn.Module):

    def __init__(self, n_embd, n_head, blocksize, LCompression, flash=False):
        super().__init__()
        self.d_head = n_embd // n_head
        self.LComp = LCompression

        self.c_attn = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_head * LCompression, n_embd)
        

        self.n_head = n_head
        self.n_embd = n_embd
        self.latent = nn.Linear(n_embd, LCompression)
        self.Wd = nn.Linear(n_embd, n_head * LCompression)

        
        self.flash = flash
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(blocksize, blocksize))
                                        .view(1, 1, blocksize, blocksize))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.latent(x).unsqueeze(2).expand([-1, -1, self.n_head, self.LComp]).transpose(1, 2) # (B, nh, T, hs)
        q = self.Wd(x).view(B, T, self.n_head, self.LComp).transpose(1, 2) # (B, nh, T, hs)
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, k, attn_mask=None, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.unsqueeze(1).transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ k.unsqueeze(1) # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.LComp) # re-assemble all head outputs side by side

        y = self.c_proj(y)
        return y




class Block(nn.Module):
    def __init__(self, n_embd, n_head, LCompression, blocksize, flashATTN):
        super().__init__()

        self.atten = CausalSelfAttentionMLA(n_embd, n_head, blocksize, LCompression, flash=flashATTN)
        self.ffwd = mlp(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.atten(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class mlp(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.act = nn.GELU()
        self.L1 = nn.Linear(n_embd, n_embd*4)
        self.L2 = nn.Linear(n_embd*4, n_embd)

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        return x


class Model(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, blockSize, LCompression = (576//2), flashATTN = True):
        super().__init__()
        self.Layers = nn.ModuleList([Block(n_embd, n_head, LCompression, blockSize, flashATTN) for i in range(n_layer)])
        self.tokEmb = nn.Embedding(vocab_size, n_embd)
        self.posEmb = nn.Embedding(blockSize, n_embd)
        self.lmHead = nn.Linear(n_embd, vocab_size)
        self.lnf = nn.LayerNorm(n_embd)

    def forward(self, x, target = None):
        b,t = x.size()
        x = self.tokEmb(x)
        pos_emb = self.posEmb(torch.arange(t, device=x.device))
        x = x + pos_emb
        for i in range(len(self.Layers)):
            x = self.Layers[i](x)
        x = self.lnf(x)
        x = self.lmHead(x)

        if(target != None):
            b,t,c = x.size()
            ins = torch.reshape(x, (b*t, c))
            loss = nn.functional.cross_entropy(input=ins, target=target.view(-1))
            return x, loss
        return x
    
if __name__ == "__main__":
    os.system('cls')
    testhead = Model(4,32,2,128, 3000).to("cuda")
    x = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    y = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    out = testhead(x, y)

    print(out)