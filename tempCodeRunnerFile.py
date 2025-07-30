import torch
import torch.distributions.multivariate_normal as dbs

M = torch.rand([32,64], dtype=torch.float32)
dim = 0
s = torch.zeros(dim)
fn = dbs.MultivariateNormal(s, torch.eye(dim))#.sample()
omega = 8
FNS = [torch.cos(),torch.sin()]

def Aprox_softmax(x):
    w = torch.tensor([fn.sample(x.size()) for _ in range(w)], dtype=torch.float32)
    R = torch.tensor([ FNS[min(i/omega)](w[i].T()*x) for i in range(w*FNS)])

    return x