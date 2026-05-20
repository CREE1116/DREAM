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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        
        self.norm1 = RMSNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.rope = RotaryPositionalEmbedding(d_model // n_heads)
        self.q_norm = RMSNorm(d_model // n_heads)
        self.k_norm = RMSNorm(d_model // n_heads)
        
        self.norm2 = RMSNorm(d_model)
        d_ff = int(d_model * 8 / 3)
        self.ffn = nn.ModuleDict({
            "w1": nn.Linear(d_model, d_ff, bias=False),
            "w2": nn.Linear(d_model, d_ff, bias=False),
            "w3": nn.Linear(d_ff, d_model, bias=False),
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, step_vec_head, is_causal=True, freeze_mask_ste=None):
        batch_size, seq_len, _ = x.size()
        
        x_norm = self.norm1(x)
        q_proj = self.W_q(x_norm)
        k_proj = self.W_k(x_norm)
        v_proj = self.W_v(x_norm)

        q = q_proj.view(batch_size, seq_len, self.n_heads, -1)
        k = k_proj.view(batch_size, seq_len, self.n_heads, -1)
        v = v_proj.view(batch_size, seq_len, self.n_heads, -1)

        cos, sin = self.rope(q)
        q, k = apply_rope(q, k, cos, sin)

        # 간섭 방지를 위해 RoPE 적용 후 스텝 임베딩 더하기
        q = q + step_vec_head
        k = k + step_vec_head
        v = v + step_vec_head

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.W_o(out)
        out = self.dropout(out)

        # Freeze Mask 적용: 얼어붙은 토큰은 업데이트 안됨 (하지만 위에서 K, V로는 쓰였음!)
        if freeze_mask_ste is not None:
            out = out * (1.0 - freeze_mask_ste)
        
        x_attn = x + out

        x_norm2 = self.norm2(x_attn)
        ffn_out = self.ffn["w3"](F.silu(self.ffn["w1"](x_norm2)) * self.ffn["w2"](x_norm2))
        ffn_out = self.dropout(ffn_out)

        if freeze_mask_ste is not None:
            ffn_out = ffn_out * (1.0 - freeze_mask_ste)
            
        return x_attn + ffn_out

class DREAM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, max_steps=64,
                 tau=0.98, min_steps=8, freeze_dropout=0.2, n_blocks=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.tau = tau
        self.freeze_dropout = freeze_dropout
        self.n_blocks = n_blocks

        # 임베딩 레이어
        self.E_s = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.E_s.weight, mean=0.0, std=d_model ** -0.5)

        # Step sinusoidal (고정) - 러너블을 피하기 위해 원본 코사인 복원
        # freeze_ratio와 step 진행도 2가지를 결합하기 위해 d_model // 4 크기 사용
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 4).float() / d_model))
        self.register_buffer("step_inv_freq", inv_freq)
        self.step_scale = nn.Parameter(torch.tensor(0.02))

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])

        self.final_norm = RMSNorm(d_model)
        self.fc_head = nn.Linear(d_model, vocab_size, bias=False)
        self.fc_head.weight = self.E_s.weight

    def _step_vector(self, freeze_ratio, step, device):
        # [수렴 비율, 현재 스텝 진행도] 두 가지 정보를 결합
        progress = torch.tensor([freeze_ratio, step / self.max_steps], device=device).float() # (2,)
        inp = progress.unsqueeze(-1) * self.step_inv_freq # (2, d_model // 4)
        
        # sin, cos 결합 후 flatten하여 (d_model,) 크기 생성
        vec = torch.cat([torch.sin(inp), torch.cos(inp)], dim=-1).flatten()
        return vec * self.step_scale

    def forward(self, tokens, is_causal=True):
        E_d = self.E_s(tokens)
        batch_size, seq_len, _ = E_d.size()
        device = E_d.device

        # 상태 추적용 변수들
        is_frozen = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool, device=device)
        total_steps_accum = torch.zeros(batch_size, seq_len, device=device)
        freeze_ratio_accum = torch.tensor(0.0, device=device)

        for step in range(1, self.max_steps + 1):
            prev_E_d = E_d.clone()
            
            # 얼어붙은 토큰은 연산 결과를 0으로 만들어 E_d를 그대로 유지함
            freeze_mask_float = is_frozen.float()

            # 측정용 지표 업데이트
            total_steps_accum = total_steps_accum + (~is_frozen.squeeze(-1)).float()
            freeze_ratio_accum = freeze_ratio_accum + is_frozen.float().mean()

            # Block Processing
            freeze_ratio = is_frozen.float().mean()
            step_vec = self._step_vector(freeze_ratio, step, device)
            step_vec_head = step_vec.view(1, 1, self.n_heads, -1)

            for block in self.blocks:
                E_d = block(E_d, step_vec_head, is_causal=is_causal, freeze_mask_ste=freeze_mask_float)

            # Cosine Similarity 기반 수렴(Halting) 체크
            with torch.no_grad():
                cos_sim = F.cosine_similarity(E_d, prev_E_d, dim=-1)
                new_frozen = (cos_sim > self.tau).unsqueeze(-1) & (~is_frozen)

                if step < self.min_steps:
                    new_frozen = torch.zeros_like(new_frozen)

                if self.training and self.freeze_dropout > 0:
                    keep_pondering = torch.rand_like(cos_sim).unsqueeze(-1) < self.freeze_dropout
                    new_frozen = new_frozen & (~keep_pondering)

                is_frozen = is_frozen | new_frozen

            if is_frozen.all():
                break

        self.last_steps = total_steps_accum.mean().item()
        self.last_freeze_ratio = (freeze_ratio_accum / step).item()
        self.last_ponder_steps = total_steps_accum 

        return self.fc_head(self.final_norm(E_d))
