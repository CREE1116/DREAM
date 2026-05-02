import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_period=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_period = max_period

    def forward(self, x):
        batch_size, seq_len, n_heads, head_dim = x.shape
        device = x.device

        position = torch.arange(seq_len, device=device).unsqueeze(1)
        dim_t = torch.arange(0, head_dim, 2, device=device) / head_dim
        inv_freq = 1.0 / (self.max_period ** dim_t)

        sinusoid_inp = position * inv_freq
        sin = torch.sin(sinusoid_inp).repeat_interleave(2, dim=-1).view(1, seq_len, 1, head_dim)
        cos = torch.cos(sinusoid_inp).repeat_interleave(2, dim=-1).view(1, seq_len, 1, head_dim)

        return cos, sin

def rotate(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)

class DREAM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, max_steps=64,
                 tau=0.98, min_steps=8, freeze_dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_steps = max_steps
        self.tau = tau
        self.min_steps = min_steps
        self.freeze_dropout = freeze_dropout

        # Embedding
        self.E_s = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.E_s.weight, mean=0.0, std=d_model ** -0.5)

        # Step sinusoidal (고정)
        # freeze_ratio와 step 진행도 2가지를 결합하기 위해 d_model // 4 크기 사용
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 4).float() / d_model))
        self.register_buffer("step_inv_freq", inv_freq)
        self.step_scale = nn.Parameter(torch.tensor(0.02))

        # Attention
        self.rope = RotaryPositionalEmbedding(d_model // n_heads)
        self.q_norm = RMSNorm(d_model // n_heads)
        self.k_norm = RMSNorm(d_model // n_heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # SwiGLU FFN
        d_ff = int(d_model * 8 / 3)
        self.ffn = nn.ModuleDict({
            "w1": nn.Linear(d_model, d_ff, bias=False),
            "w2": nn.Linear(d_model, d_ff, bias=False),
            "w3": nn.Linear(d_ff, d_model, bias=False),
        })

        self.final_norm = RMSNorm(d_model)
        self.fc_head = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_head.weight = self.E_s.weight  # Weight Tying

    def _step_vector(self, freeze_ratio, step, device):
        # [수렴 비율, 현재 스텝 진행도] 두 가지 정보를 결합
        progress = torch.tensor([freeze_ratio, step / self.max_steps], device=device).float() # (2,)
        inp = progress.unsqueeze(-1) * self.step_inv_freq # (2, d_model // 4)
        
        # sin, cos 결합 후 flatten하여 (d_model,) 크기 생성
        vec = torch.cat([torch.sin(inp), torch.cos(inp)], dim=-1).flatten()
        return vec * self.step_scale

    def evolution_step(self, E_d, freeze_ratio, step, is_causal=True, freeze_mask=None):
        batch_size, seq_len, _ = E_d.size()
        device = E_d.device

        step_vec = self._step_vector(freeze_ratio, step, device)

        x_norm = self.norm1(E_d)
        q_proj = self.W_q(x_norm) + step_vec
        k_proj = self.W_k(x_norm) + step_vec
        v_proj = self.W_v(x_norm)

        q = q_proj.view(batch_size, seq_len, self.n_heads, -1)
        k = k_proj.view(batch_size, seq_len, self.n_heads, -1)
        v = v_proj.view(batch_size, seq_len, self.n_heads, -1)

        # QK Norm 적용 (학습 가능한 가중치 포함)
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.W_o(out)

        if freeze_mask is not None:
            out = out.masked_fill(freeze_mask.unsqueeze(-1), 0.0)
        E_attn = E_d + out

        x_norm2 = self.norm2(E_attn)
        ffn_out = self.ffn["w3"](F.silu(self.ffn["w1"](x_norm2)) * self.ffn["w2"](x_norm2))

        if freeze_mask is not None:
            ffn_out = ffn_out.masked_fill(freeze_mask.unsqueeze(-1), 0.0)
        E_new = E_attn + ffn_out

        return E_new

    def forward(self, tokens, is_causal=True):
        E_d = self.E_s(tokens)
        batch_size, seq_len, _ = E_d.size()

        converged = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=E_d.device)
        total_steps_accum = torch.zeros(batch_size, seq_len, device=E_d.device)
        freeze_ratio_accum = torch.tensor(0.0, device=E_d.device)

        for step in range(1, self.max_steps + 1):
            prev_E_d = E_d.clone()
            freeze_ratio = converged.float().mean()
            E_d = self.evolution_step(E_d, freeze_ratio, step, is_causal=is_causal, freeze_mask=converged)

            with torch.no_grad():
                cos_sim = F.cosine_similarity(E_d, prev_E_d, dim=-1)
                newly_converged = (cos_sim > self.tau) & (~converged)

                if step < self.min_steps:
                    newly_converged = torch.zeros_like(newly_converged)

                if self.training and self.freeze_dropout > 0:
                    keep_pondering = torch.rand_like(cos_sim) < self.freeze_dropout
                    newly_converged = newly_converged & (~keep_pondering)

                converged = converged | newly_converged
                total_steps_accum = total_steps_accum + (~converged).float()
                freeze_ratio_accum = freeze_ratio_accum + converged.float().mean()

            if converged.all():
                break

        self.last_steps = total_steps_accum.mean().item()
        self.last_freeze_ratio = (freeze_ratio_accum / step).item()
        self.last_ponder_steps = total_steps_accum

        return self.fc_head(self.final_norm(E_d))
