import torch

M = torch.rand([32,64], dtype=torch.float32)


def Aprox_softmax(x):

    return x



msm = torch.softmax(M)
masm = Aprox_softmax(M)

if(torch.allclose(msm, masm, rtol=1e-2, atol=1e-2)):
    print("equal")
else:
    print("not equal")



