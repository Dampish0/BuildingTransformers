import torch
from torch import nn
import math
import os
torch.manual_seed(42)

LCompression = 576//2

class Head(nn.Module):
    def __init__(self, n_embd, n_head, blocksize, masked=False):
        super().__init__()
        self.d_head = n_embd // n_head
        self.n_embd = n_embd
        self.masked = masked

        # self.weightQ = nn.Linear(n_embd, self.d_head)
        self.Wd = nn.Linear(n_embd, LCompression)
        # self.Wuk = nn.Linear(LCompression, self.d_head)
        # self.weightK = nn.Linear(n_embd,self.d_head)
        # self.weightV = nn.Linear(n_embd,self.d_head)
        if masked:
            self.register_buffer("mask", (torch.tril(torch.ones([blocksize, blocksize]), diagonal=0).unsqueeze(0) == 0))

    def forward(self, x, Wdkv):
        b,t,c = x.size()
        
        # x: t x n_embd
        
        # Wdk: t 

        # print((torch.transpose(self.Wuk(Wdkv), 1,2)).shape, self.Wuk(Wdkv).shape, self.weightQ(x).shape, x.shape)
        
        # R = self.weightQ.weight @ (torch.transpose(self.Wuk.weight, 0, 1))
        # exit()
        # # R = WeightQ @ Wuk^T  =>  (n_embd x d_Head)  @ (LC x d_head)T  =>  (n_embd x d_Head)  @ (d_head x LC)  =>  n_embd x L_compression, which is correct
        # # X @ R  =>  (t x n_embd) @ (n_embd x L_compression)  => (t x L_compression), voila we now have our new Q


        # q = self.weightQ(x) @ (torch.transpose(self.Wuk(Wdkv), 1,2))
        
        # # desired outcome for q' is: (t x LCompression)


        # print(self.Wd(x).shape, Wdkv.shape, (torch.transpose(Wdkv, 1,2)).shape)
        #exit()
        h = self.Wd(x) @ (torch.transpose(Wdkv, 1,2))
        # print(h.shape)
        h = h / math.sqrt(self.d_head)

        if(self.masked):
            #forgot masking
            h = h.masked_fill(self.mask[:, :t, :t], float('-inf'))

        y = torch.softmax(h, dim=-1)

        y = y @ Wdkv

        return y
    




class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, blocksize, masked=False):
        super().__init__()       
        self.d_head = n_embd // n_head
        self.n_embd = n_embd

        self.heads = nn.ModuleList([Head(n_embd, n_head, blocksize, masked) for i in range(n_head)])
        self.latent = nn.Linear(n_embd, LCompression)
        # self.weight0 = nn.Linear(LCompression, n_embd)
        # self.Wuv = nn.Linear(LCompression, self.d_head)

        self.outW = nn.Linear(n_head * LCompression, n_embd)

    def forward(self, x):
        # b, t, c
        Wdkv = self.latent(x)
        
        z = [self.heads[i](x, Wdkv) for i in range(len(self.heads))]
        z = torch.concat(z, dim=2)
        # print(z.shape)
        # A = self.Wuv.weight @ self.weight0.weight
        # k = A(z)
        # exit()
        return self.outW(z)
    


class Block(nn.Module):
    def __init__(self, n_embd, n_head, blocksize):
        super().__init__()

        self.atten = MultiHeadAttention(n_embd, n_head, blocksize, masked=True)
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
    def __init__(self, n_layer, n_embd, n_head, vocab_size, blockSize):
        super().__init__()
        self.Layers = nn.ModuleList([Block(n_embd, n_head, blockSize) for i in range(n_layer)])
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