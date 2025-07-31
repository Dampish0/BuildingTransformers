import torch
import math
import torch.distributions.multivariate_normal as dbs
import torch.nn as nn

dim = 32
# M = torch.rand([dim], dtype=torch.float32).unsqueeze(0)

s = torch.zeros(dim)
fn = dbs.MultivariateNormal(s, torch.eye(dim))#.sample()

nmr = math.sqrt(dim)*s


omega = 4

def Aprox_softmax(k, q):
    
    w = torch.randn(omega, dim, device=k.device).t()

    # regularize
    w = math.sqrt(dim) * (w/torch.norm(w))

    kNorm = torch.norm(k)
    qNorm = torch.norm(q)

    z = k + q
    zSqr = kNorm**2 + qNorm**2

    print(w.shape, z.shape, k.shape, q.shape)
    Lcosh = torch.cosh(torch.matmul(z, w))
    A = torch.exp(-zSqr/2)
    
    R = A * Lcosh.mean(dim=-1)
    





    return R

class Head(nn.Module):
    def __init__(self, n_embd, n_head, blocksize):
        super().__init__()
        self.d_head = n_embd // n_head
        self.n_embd = n_embd

        self.weightQ = nn.Linear(n_embd,self.d_head)
        self.weightK = nn.Linear(n_embd,self.d_head)
        self.weightV = nn.Linear(n_embd,self.d_head)

    def forward(self, x):
        
        k = self.weightK(x) # n_embd, d_head
        q = self.weightQ(x) # n_embd, d_head
        v = self.weightV(x) # n_embd, d_head

        Aprox_softmax(k,q)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, blocksize):
        super().__init__()       
        self.n_head = n_head
        self.heads = nn.ModuleList([Head(n_embd, n_head, blocksize) for i in range(n_head)])
        self.weight0 = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # b, t, c
        z = [self.heads[i](x) for i in range(len(self.heads))]
        z = torch.concat(z, dim=2)
        

        k = self.weight0(z)
        return k
    


class Block(nn.Module):
    def __init__(self, n_embd, n_head, blocksize):
        super().__init__()

        self.atten = MultiHeadAttention(n_embd, n_head, blocksize)
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
    testhead = Model(4,dim,2,128, 3000).to("cuda")
    x = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    y = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    out = testhead(x, y)

    print(out)


