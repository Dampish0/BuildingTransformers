import torch
import math
import torch.distributions.multivariate_normal as dbs

M = torch.rand([32], dtype=torch.float32).unsqueeze(1)
dim = 32
s = torch.zeros(dim)
fn = dbs.MultivariateNormal(s, torch.eye(dim))#.sample()
omega = 8

FNS = [torch.cos,torch.sin]

def Aprox_softmax(x):
    w = [fn.sample(x.size()) for _ in range(omega)]
    print(w[0].shape, x.shape)

    

    R = torch.concat([FNS[math.floor(int(i/omega))](torch.transpose(w[i%omega], 0, 2) @ x) for i in range(omega*len(FNS))], dim=0)
    R = 1 * R
    R = R / math.sqrt(omega)
    print(R.shape)
    return R



msm = torch.softmax(M, dim=-1)
masm = Aprox_softmax(M)

print(msm.shape, masm.shape)

if(torch.allclose(msm, masm, rtol=1e-2, atol=1e-2)):
    print("equal")
else:
    print("not equal")



