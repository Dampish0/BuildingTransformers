# Trying to learn math behind attention and learn how to create model architectures
This project is a hands-on exploration of transformer architectures, focusing on understanding the math behind attention mechanisms and building models from scratch in PyTorch.    

## Current Features
- Multi-Head Attention
- DeepSeek MLA
- Custom extremely basic training loop, i can make it better but i only use this loop to test out the models.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Goals & Progress](#goals--progress)
- [Findings](#findings)


## Results
All models get the same amount of training steps and same data, the only exception are their model specific parameters for example, LCompression in MLA.  
  
MultiHeadAttention:
<img width="1274" height="412" alt="image" src="https://github.com/user-attachments/assets/2e738f78-34ae-40b2-b9a2-c34608f8056c" />
*Figure 1: MultiHeadAttention loss over time along with time per step aswell with a final output to show the generation of the model*

MLA (LCompression = 576):  
<img width="769" height="195" alt="image" src="https://github.com/user-attachments/assets/4ddf061f-78ed-4c00-9ff0-caaaaddbbc77" />
*Figure 2: MLA using the same ammount of Latent space as R1 with loss over time along with time per step aswell with a final output to show the generation of the model*

MLA (LCompression = 288 = 576/2):  
<img width="740" height="425" alt="image" src="https://github.com/user-attachments/assets/1b616ec2-f665-44e8-9bfa-58cae971ae28" />
*Figure 3: MLA using half of Latent space as R1 with even lower loss. I also include time per step aswell with a final output to show the generation of the model*

## Goals & Progress

- ✅ Create a basic transformer model

- ✅ Implement deepseek MLA and test it out

- ❌ decoder and encoder layers and try to train it to make sure it works.

- ❌ Try to make my own optimizer or just write SGD by manually calculating the gradients, meaning no AutoGrad. Basically doing my own backPropogation

- ❌ try to implement performers according to [https://arxiv.org/abs/2009.14794]

- ❌ Creating my own sub-word tokenizer.

## Usage
You just have to create an instance of the model from the specific .py file, for example i have MLA.py, model.py and in both of them there is a class named "Model" just instantiate with  
```py
model = Model(n_layers,n_embd,n_head,vocab_size,blocksize)
```  
It is done this way to make things easy, then its just normal training like any other model. 
> [!WARNING]  
> I have not added in the scaled dot product kernel which makes things alot faster, i can do this easily but im only testing and learning here so its not something im looking to do. If someone requests it i will make a new branch and add it in for them.
  
  
## Installation
Just clone it and make sure you have pytorch installed and everything should run fine.  
```bash
git clone https://github.com/yourusername/transformers_learning.git
cd transformers_learning/AttentionIsAllYouNeed
```  
  
## Findings
From my testing, MLA takes more vram and is larger in parameters for the same hyperparameters as MHA, however it does seem to generate lower loss in same amount of training steps which can be seen in the uploaded images. The vram usage is trippled but we do get slightly lower loss. Either i have made an error in the implementation or the LCompression variable in MLA.py is too high.

When i tried to train with half of the latent compression i got even lower loss which is probably because a higher latent space might need longer training to learn how to represent properly.

For the same hyperparameters in the models we get  
MLA: 80.0M parameters  
MHA: 21.0M parameters  
  
When i halve the latent compression (LCompression) hyperparamter in the top of the MLA.py file. (LCompression = 576/2 = 288)  
MLA: 47.0M parameters  
MHA: 21.0M parameters  
