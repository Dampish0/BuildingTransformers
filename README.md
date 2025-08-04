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
MultiHeadAttention:
<img width="1274" height="412" alt="image" src="https://github.com/user-attachments/assets/2e738f78-34ae-40b2-b9a2-c34608f8056c" />

MLA:  
<img width="770" height="302" alt="image" src="https://github.com/user-attachments/assets/28f4c3ae-0ef0-4811-9fb0-2543d46dbc60" />

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