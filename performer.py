import torch
import math
import torch.distributions.multivariate_normal as dbs
import torch.nn as nn

M = torch.rand([32], dtype=torch.float32).unsqueeze(0)
dim = 32
s = torch.zeros(dim)
fn = dbs.MultivariateNormal(s, torch.eye(dim))#.sample()
omega = 8

FNS = [torch.cos,torch.sin]

def Aprox_softmax(k, q):
    w = torch.concat([fn.sample(k.size()) for _ in range(omega)])
    print(w.shape)
    kNorm = torch.norm(k)
    qNorm = torch.norm(q)
    z = k + q
    zSqr = torch.pow(kNorm, 2) +  torch.pow( qNorm, 2)

    A = torch.exp(-zSqr/2)
    cosH = torch.cosh(w.transpose(1,2) @ z)

    R = (A @ w) * cosH

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

hd = Head(32,4,128,False)
hd(M)

# msm = torch.softmax(M, dim=-1)
# masm = Aprox_softmax(M)

# print(msm.shape, masm.shape)

# if(torch.allclose(msm, masm, rtol=1e-2, atol=1e-2)):
#     print("equal")
# else:
#     print("not equal")



