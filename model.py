import torch
from torch import nn
import math
torch.manual_seed(42)



class Head(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.d_head = n_embd // n_head

        self.weightQ = nn.Linear(n_embd,n_embd)
        self.weightK = nn.Linear(n_embd,n_embd)
        self.weightV = nn.Linear(n_embd,n_embd)
        

    def forward(self, x):
        k = self.weightK(x)
        q = self.weightQ(x)
        v = self.weightV(x)
        
        h = q @ (torch.transpose(k, 1,2))
        h = h / math.sqrt(self.d_head)

        y = torch.softmax(h, dim=-1)

        y = y @ v


        return y
    




class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()       
        self.n_head = n_head
        self.heads = nn.ModuleList([Head(n_embd, n_head) for i in range(n_head)])
        self.weight0 = nn.Linear(n_embd*n_head, n_embd)

    def forward(self, x):

        z = self.heads[0](x)
        for i in range(self.n_head-1):
           z = torch.concat([z, self.heads[i+1](x)], dim=2)

        k = self.weight0(z)
        return k
    


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.atten = MultiHeadAttention(n_embd, n_head)
        self.ffwd = mlp(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)


    def forward(self, x):
        x = x + self.ln1(self.atten(x))
        x = x + self.ln2(self.ffwd(x))
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
    def __init__(self, n_layer, n_embd, n_head, vocab_size):
        super().__init__()
        self.Layers = nn.ModuleList([Block(n_embd, n_head) for i in range(n_layer)])
        self.tokEmb = nn.Embedding(vocab_size, n_embd)
        self.lmHead = nn.Linear(n_embd, vocab_size)

    def forward(self, x, target = None):
        
        x = self.tokEmb(x)

        for i in range(len(self.Layers)):
            x = self.Layers[i](x)
        
        x = self.lmHead(x)

        if(target != None):
            b,t,c = x.size()
            ins = torch.reshape(x, (b*t, c))
            loss = nn.functional.cross_entropy(input=ins, target=target.view(-1))
            return x, loss
        return x

# testhead = Model(4,32,2,128).to("cuda")
# x = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
# y = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
# out = testhead(x, y)

# print(out)