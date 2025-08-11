import torch
from torch import nn
import math
import os
torch.manual_seed(42)

# a fully speced out model architecture that's ready to go. It has many of the latest features, MOE maybe coming soon
    
import torch.nn.functional as F

class CausalSelfAttentionMLA(nn.Module):

    def __init__(self, n_embd, n_head, blocksize, LCompression, flash=False, attn_dropout=0.0):
        super().__init__()
        self.d_head = n_embd // n_head
        self.LComp = LCompression
        assert self.LComp % 2 == 0, "RoPE requires an even head dimension (LCompression)."
        self.rope_theta = 10000.0  # standard RoPE base

        #self.c_attn = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_head * LCompression, n_embd)
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.latent = nn.Linear(n_embd, LCompression)
        self.Wd = nn.Linear(n_embd, n_head * LCompression)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.flash = flash
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(blocksize, blocksize))
                                        .view(1, 1, blocksize, blocksize))

    # RoPE helpers
    @staticmethod
    def _rotate_half(x):
        # x: (..., d)
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        # stack [-x_odd, x_even] interleaved
        return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

    def _apply_rope(self, x, start_index: int = 0):
        # x: (B, nh, T, d), apply RoPE along last dim with T positions starting at start_index
        B, nh, T, d = x.shape
        assert d % 2 == 0
        device = x.device
        dtype_x = x.dtype
        dtype_rope = torch.float32 if dtype_x in (torch.float16, torch.bfloat16) else dtype_x

        half = d // 2
        # positions
        pos = torch.arange(start_index, start_index + T, device=device, dtype=dtype_rope)  # (T,)
        # frequencies: base^(-2i/d) with i = 0..half-1
        inv_freq = torch.pow(torch.tensor(self.rope_theta, device=device, dtype=dtype_rope),
                             -torch.arange(0, half, device=device, dtype=dtype_rope) / half)  # (half,)
        angles = pos[:, None] * inv_freq[None, :]  # (T, half)
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)[None, None, :, :].to(dtype_x)  # (1,1,T,d)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)[None, None, :, :].to(dtype_x)  # (1,1,T,d)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x, past_k_base=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # MLA: Wudv is shared base across heads (keys=values), q is per-head projection
        Wudv_curr = self.latent(x).unsqueeze(2).expand([-1, -1, self.n_head, self.LComp]).transpose(1, 2) # (B, nh, T, Lc)
        q = self.Wd(x).view(B, T, self.n_head, self.LComp).transpose(1, 2) # (B, nh, T, Lc)

        if use_cache:
            # Concatenate past base with current
            if past_k_base is not None:
                k_base = torch.cat([past_k_base, Wudv_curr], dim=2)  # (B, nh, S_total, Lc)
                S_past = past_k_base.size(2)
            else:
                k_base = Wudv_curr
                S_past = 0
            v = k_base  # values remain unrotated
            S_total = k_base.size(2)

            # Apply RoPE: q uses positions [S_past..S_total-1], k uses [0..S_total-1]
            q_rot = self._apply_rope(q, start_index=S_past)                  # (B, nh, T, Lc)
            k_rot = self._apply_rope(k_base, start_index=0)                  # (B, nh, S_total, Lc)

            if self.flash:
                # If T > 1, mask only the future within the current chunk; past is fully visible
                attn_mask = None
                if T > 1:
                    device = x.device
                    # future mask over current chunk (T x T), True means masked
                    future = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
                    if S_past > 0:
                        left = torch.zeros(T, S_past, dtype=torch.bool, device=device)
                        attn_mask = torch.cat([left, future], dim=1)  # (T, S_total)
                    else:
                        attn_mask = future
                y = F.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=attn_mask, is_causal=False)
            else:
                att = (q_rot @ k_rot.transpose(-2, -1)) * (1.0 / math.sqrt(k_rot.size(-1)))  # (B, nh, T, S_total)
                if T > 1:
                    device = x.device
                    future = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
                    if S_past > 0:
                        left = torch.zeros(T, S_past, dtype=torch.bool, device=device)
                        mask = torch.cat([left, future], dim=1)  # (T, S_total)
                    else:
                        mask = future
                    att = att.masked_fill(mask.view(1, 1, T, S_total), float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, Lc)
        else:
            # Full-seq causal attention (no cache)
            k_base = Wudv_curr
            v = k_base
            # Apply RoPE with positions [0..T-1]
            q_rot = self._apply_rope(q, start_index=0)
            k_rot = self._apply_rope(k_base, start_index=0)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = F.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=None, is_causal=True)
            else:
                # manual implementation of attention
                att = (q_rot @ k_rot.transpose(-2, -1)) * (1.0 / math.sqrt(k_rot.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, Lc)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.LComp) # re-assemble all head outputs side by side
        y = self.c_proj(y)

        if use_cache:
            return y, k_base  # cache unrotated base; RoPE reapplied each step
        return y

class CausalSelfAttention_TSSA(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout=0.3):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.c_attn = nn.Linear(n_embd, n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.temp = nn.Parameter(torch.ones(n_head,1))
        self.denom_bias = nn.Parameter(torch.zeros(n_head, block_size,1))
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        w = self.c_attn(x)

        w = w.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        w_sq = w ** 2
        denom = (torch.cumsum(w_sq,dim=-2)).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias[:,:T,:]
        tmp = torch.sum(w_normed, dim=-1)* self.temp
        
        Pi = F.softmax(tmp, dim=1) # B, nh, T
        dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=-2) / (Pi.cumsum(dim=-1) + 1e-8).unsqueeze(-1)
        attn = 1. / (1 + dots)       
        attn = self.attn_dropout(attn)
        y = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, n_embd, n_head, LCompression, blocksize, flashATTN, attn_dropout=0.3):
        super().__init__()
        #self.atten = CausalSelfAttention_TSSA(n_embd, n_head, blocksize, dropout=attn_dropout)
        self.atten = CausalSelfAttentionMLA(n_embd, n_head, blocksize, LCompression, flash=flashATTN, attn_dropout=attn_dropout)
        self.ffwd = swiglu(n_embd)
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.atten(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def forward_with_cache(self, x, cache=None):
        # cache is a dict with key: 'k_base'
        k_base = cache['k_base'] if cache is not None and 'k_base' in cache else None

        y, k_base_new = self.atten(self.ln1(x), past_k_base=k_base, use_cache=True)
        x = x + y
        x = x + self.ffwd(self.ln2(x))
        return x, {'k_base': k_base_new}

class swiglu(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # SwiGLU: inner ≈ 8/3 * n_embd (matches previous scale ~2.67)
        inner = int(2 * n_embd)
        self.w_gate = nn.Linear(n_embd, inner)
        self.w_up   = nn.Linear(n_embd, inner)
        self.w_down = nn.Linear(inner, n_embd)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.w_gate(x)) * self.w_up(x)
        x = self.w_down(x)
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

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)
    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = F.pad(x, (self.kernel_size - 1, 0))      # pad left only
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C_out)
        return x
    
class Model(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, vocab_size, blockSize, LCompression = (576//2), flashATTN = True, attn_dropout=0.3):
        super().__init__()
        self.Layers = nn.ModuleList([Block(n_embd, n_head, LCompression, blockSize, flashATTN, attn_dropout=attn_dropout) for i in range(n_layer)])
        self.ConvL = nn.Sequential(
                                    CausalConv1d(n_embd, n_embd, 9),
                                    nn.GELU(),
                                    nn.RMSNorm(n_embd),
                                    CausalConv1d(n_embd, n_embd, 6),
                                    nn.GELU(),
                                    nn.RMSNorm(n_embd),
                                    CausalConv1d(n_embd, n_embd, 3),
                                    nn.GELU(),
                                    nn.RMSNorm(n_embd)
                                   )

        self.tokEmb = nn.Embedding(vocab_size, n_embd)
        #self.wpe = nn.Embedding(blockSize, n_embd)
        # removed absolute positional embeddings; RoPE is applied inside attention
        self.lmHead = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie weights
        self.lmHead.weight = self.tokEmb.weight
        self.lnf = nn.RMSNorm(n_embd)
        # --- custom initialization to stabilize tied weights ---
        self.apply(self._init_weights)
        self._post_init_scale()

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

    def _post_init_scale(self):
        # Scale residual branch output projections to reduce initial logit variance
        # Common practice: 1/sqrt(2 * n_layers)
        scale = 1.0 / math.sqrt(2 * len(self.Layers))
        with torch.no_grad():
            for layer in self.Layers:
                # attention out proj
                layer.atten.c_proj.weight.mul_(scale)
                # mlp down proj
                layer.ffwd.w_down.weight.mul_(scale)

    def forward(self, x, target = None):
        b,t = x.size()
        te = self.tokEmb(x)
        #pos_emb = self.wpe(torch.arange(0, t, dtype=torch.long, device=x.device)) # position embeddings of shape (t, n_embd)
        x = (te + pos_emb)
        # RoPE replaces absolute positions; no pos_emb addition

        x = x + self.ConvL(x)

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
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        Args:
          input_ids: (B, T) LongTensor of token ids.
          max_new_tokens: number of tokens to generate.
          do_sample: if False, greedy decoding; if True, sample with top-k/top-p.
          temperature: sampling temperature; ignored if do_sample is False or <= 0.
          top_k: keep only top k tokens for sampling.
          top_p: nucleus sampling threshold in (0,1]; set None or 1.0 to disable.
          eos_token_id: stop when generated; if None, generate full max_new_tokens.
          use_cache: use MLA key/value caching for fast decoding.
        Returns:
          (B, T + generated) tensor of token ids.
        """
        def top_k_top_p_filtering(logits: torch.Tensor, tk: int | None, tp: float | None) -> torch.Tensor:
            # logits: (B, V)
            if tk is not None and tk > 0 and tk < logits.size(-1):
                vals, _ = torch.topk(logits, tk, dim=-1)
                min_vals = vals[..., -1, None]
                logits = torch.where(logits < min_vals, torch.full_like(logits, float('-inf')), logits)
            if tp is not None and tp < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > tp
                # ensure at least one token remains
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
                filtered = torch.full_like(logits, float('-inf'))
                filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
                logits = filtered
            return logits

        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                device = input_ids.device
                seq = input_ids.clone()
                B = seq.size(0)
                finished = torch.zeros(B, dtype=torch.bool, device=device) if eos_token_id is not None else None

                # Prefill
                if use_cache:
                    x = self.tokEmb(seq)  # (B, T, C)
                    caches = [None] * len(self.Layers)
                    for i, layer in enumerate(self.Layers):
                        x, caches[i] = layer.forward_with_cache(x, cache=caches[i] or {})
                    x = self.lnf(x)
                    logits = self.lmHead(x)[:, -1, :]  # (B, V)
                else:
                    caches = None
                    logits = self.forward(seq)[:, -1, :]  # (B, V)

                for _ in range(max_new_tokens):
                    # Select next token
                    if do_sample and temperature > 0:
                        logits_scaled = logits / max(temperature, 1e-8)
                        logits_f = top_k_top_p_filtering(logits_scaled, top_k, top_p)
                        probs = torch.softmax(logits_f, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)

                    # Respect finished sequences
                    if eos_token_id is not None:
                        eos_col = torch.full_like(next_token, eos_token_id)
                        next_token = torch.where(finished.view(-1, 1), eos_col, next_token)

                    # Append
                    seq = torch.cat([seq, next_token], dim=1)

                    # Update finished and maybe early stop
                    if eos_token_id is not None:
                        finished = finished | (next_token.view(-1) == eos_token_id)
                        if torch.all(finished):
                            break

                    # Compute next logits
                    if use_cache:
                        x = self.tokEmb(next_token)  # (B, 1, C)
                        for i, layer in enumerate(self.Layers):
                            x, caches[i] = layer.forward_with_cache(x, cache=caches[i])
                        x = self.lnf(x)
                        logits = self.lmHead(x)[:, -1, :]
                    else:
                        logits = self.forward(seq)[:, -1, :]

                return seq
        finally:
            if was_training:
                self.train()

if __name__ == "__main__":
    os.system('cls')
    testhead = Model(4,32,2,128, 3000).to("cuda")
    x = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    y = torch.randint(low=0,high=32,size=[1,3000], dtype=torch.long, device="cuda")
    out = testhead(x, y)

    print(out)