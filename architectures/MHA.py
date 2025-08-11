import torch
from torch import nn
import math
torch.manual_seed(42)
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, n_head, blocksize, masked=False):
        super().__init__()
        self.d_head = n_embd // n_head
        self.n_embd = n_embd
        self.masked = masked

        self.weightQ = nn.Linear(n_embd,self.d_head)
        self.weightK = nn.Linear(n_embd,self.d_head)
        self.weightV = nn.Linear(n_embd,self.d_head)
        if masked:
            self.register_buffer("mask", (torch.tril(torch.ones([blocksize, blocksize]), diagonal=0).unsqueeze(0) == 0))

    def forward(self, x):
        b,t,c = x.size()
        # x = b, t , n_embd
        k = self.weightK(x) # n_embd, d_head
        q = self.weightQ(x) # n_embd, d_head
        v = self.weightV(x) # n_embd, d_head
        # print(q.shape)
        # print((torch.transpose(k, 1,2)).shape)
        h = q @ (torch.transpose(k, 1,2)) # b, t, d_head  x  b, d_head, t  =  b, t, t
        #print(h.shape)
        h = h / math.sqrt(self.d_head)

        if(self.masked):
            #forgot masking
            h = h.masked_fill(self.mask[:, :t, :t], float('-inf'))

        y = torch.softmax(h, dim=-1)

        y = y @ v


        return y
    




class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, blocksize, masked=False):
        super().__init__()       
        self.n_head = n_head
        self.heads = nn.ModuleList([Head(n_embd, n_head, blocksize, masked) for i in range(n_head)])
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
        self.L1 = nn.Linear(n_embd, int(n_embd*2.67))
        self.L2 = nn.Linear(int(n_embd*2.67), n_embd)

    def forward(self, x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        return x

class Model(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, blockSize):
        super().__init__()
        self.Layers = nn.ModuleList([Block(n_embd, n_head) for i in range(n_layer)])
        self.tokEmb = nn.Embedding(vocab_size, n_embd)
        # removed absolute positional embeddings; RoPE is applied inside attention
        self.lmHead = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie weights
        self.lmHead.weight = self.tokEmb.weight
        self.lnf = nn.RMSNorm(n_embd)
        # --- custom initialization to stabilize tied weights ---
        self.apply(self._init_weights)
        #self._post_init_scale()
        # track max context window for generation
        self.block_size = blockSize

    def _init_weights(self, module):
        # Initialize with small std; skip re-init of tied lmHead weight
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            if module is self.lmHead:
                # bias is False; weight is tied to tokEmb and already initialized
                return
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # def _post_init_scale(self):
    #     # Scale residual branch output projections to reduce initial logit variance
    #     # Common practice: 1/sqrt(2 * n_layers)
    #     scale = 1.0 / math.sqrt(2 * len(self.Layers))
    #     with torch.no_grad():
    #         for layer in self.Layers:
    #             # attention out proj
    #             layer.atten.c_proj.weight.mul_(scale)
    #             # mlp down proj
    #             layer.ffwd.w_down.weight.mul_(scale)

    def forward(self, x, target=None, past_kv=None, use_cache=False):
        # x: (B, T_in) int64
        # past_kv: Optional[List[Tuple[k, v]]] where k,v: (B, n_head, T_past, head_dim)
        b,t = x.size()
        x = self.tokEmb(x)
        presents = [] if use_cache else None
        for i in range(len(self.Layers)):
            layer_past = None
            if use_cache and past_kv is not None:
                layer_past = past_kv[i]
            if use_cache:
                x, present = self.Layers[i](x, past_kv=layer_past, use_cache=True)
                presents.append(present)
            else:
                x = self.Layers[i](x)
        x = self.lnf(x)
        x = self.lmHead(x)

        if target != None:
            b,t,c = x.size()
            ins = torch.reshape(x, (b*t, c))
            loss = F.cross_entropy(input=ins, target=target.view(-1))
            if use_cache:
                return x, loss, presents
            return x, loss
        if use_cache:
            return x, presents
        return x

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Args:
            idx: (B, T) input token ids
            max_new_tokens: number of tokens to generate
            temperature: <=0 for greedy, otherwise softmax sampling after scaling
            top_k: if set, sample only from top_k tokens
            use_cache: enable incremental decoding with KV cache
            eos_token_id: if set, stop when all sequences emit eos
        Returns:
            (B, T + max_new_tokens) with generated tokens appended
        """
        was_training = self.training
        self.eval()
        device = idx.device
        past_kv = None

        # helper for top-k filtering
        def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
            k = int(min(k, logits.size(-1)))
            if k <= 0:
                return logits
            v, ix = torch.topk(logits, k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            return mask.scatter(-1, ix, v)

        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=device) if eos_token_id is not None else None

        for step in range(max_new_tokens):
            if use_cache:
                # first step: full context (cropped), later: only last token
                if step == 0:
                    x_cond = idx[:, -self.block_size:] if self.block_size and idx.size(1) > self.block_size else idx
                else:
                    x_cond = idx[:, -1:]
                logits, present = self(x_cond, past_kv=past_kv, use_cache=True)
                past_kv = present
            else:
                # no cache: always feed full (cropped) context
                x_cond = idx[:, -self.block_size:] if self.block_size and idx.size(1) > self.block_size else idx
                logits = self(x_cond)

            logits = logits[:, -1, :]  # last time step
            if temperature is not None and temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if temperature is not None and temperature != 1.0:
                    logits = logits / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    logits = top_k_filter(logits, top_k)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                # if a sequence already finished, keep emitting eos
                if finished is not None and finished.any():
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                # update finished for those that just produced eos
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if torch.all(finished):
                    idx = torch.cat([idx, next_token], dim=1)
                    break

            idx = torch.cat([idx, next_token], dim=1)

        if was_training:
            self.train()
        return idx

if __name__ == "__main__":
    testhead = Model(4,32,2,128, 3000).to("cuda")
    x = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    y = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    out = testhead(x, y)

    print(out)