import os
import sys
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn.functional as F
from src.model import DREAM
from transformers import PreTrainedTokenizerFast
import os

def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|pad|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.bos_token = "<|startoftext|>"
    vocab_size = len(tokenizer)

    d_model, n_heads = 512, 8
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        if "config" in ckpt:
            d_model = ckpt["config"].get("d_model", d_model)
            n_heads = ckpt["config"].get("n_heads", n_heads)
            vocab_size = ckpt["config"].get("vocab_size", vocab_size)
    else:
        state_dict = None

    model = DREAM(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads).to(device)
    if state_dict: model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer

def sample_next_token(logits, input_ids, temperature=0.8, top_k=50, top_p=0.9, rep_penalty=1.1):
    # Logits shape: (batch, seq, vocab) -> 우리는 마지막 토큰만 필요 (1, vocab)
    logits = logits[0, -1, :].clone()
    
    # 1. Repetition Penalty
    # 이미 등장한 토큰들에 대해 페널티 부여
    for token_id in set(input_ids[0].tolist()):
        if logits[token_id] > 0:
            logits[token_id] /= rep_penalty
        else:
            logits[token_id] *= rep_penalty
            
    # 2. Temperature
    logits = logits / max(temperature, 1e-5)
    
    # 3. Top-K
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = -float('Inf')
        
    # 4. Top-P (Nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # 첫 번째 토큰은 무조건 포함 (최소 1개는 샘플링되어야 함)
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        
    # 5. 최종 샘플링
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).unsqueeze(0)

def generate():
    # === Inference Configuration ===
    TAU = 0.999       # Convergence threshold (Higher = deeper reasoning)
    MIN_STEPS = 1        # Minimum reasoning steps per token
    MAX_LEN = 128        # Maximum generation length
    
    # Sampling Parameters
    TEMP = 0.8           # Temperature (Creativity)
    TOP_K = 50           # Top-K filtering
    TOP_P = 0.9          # Nucleus sampling
    REP_PENALTY = 1.2    # Repetition penalty
    # ===============================

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer("checkpoints/dream_latest.pt", "checkpoints/tokenizer.json", device)
    
    # 모델 파라미터 설정 적용
    model.tau = TAU
    model.min_steps = MIN_STEPS

    print(f"\nDreamV2 Reasoning Engine — Active (tau={TAU}, min_steps={MIN_STEPS})")
    
    while True:
        try:
            prompt = input(f"\nUser > ").strip()
        except KeyboardInterrupt:
            break
            
        if not prompt: continue
        if prompt.lower() in ("exit", "q", "quit"): break

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids
        
        print("\nDREAM: ", end="", flush=True)
        with torch.no_grad():
            for i in range(MAX_LEN):
                logits = model(generated)
                
                # 설정된 파라미터로 샘플링
                next_token_tensor = sample_next_token(
                    logits, generated, 
                    temperature=TEMP, 
                    top_k=TOP_K, 
                    top_p=TOP_P, 
                    rep_penalty=REP_PENALTY
                )
                next_token_id = next_token_tensor.item()
                
                if next_token_id == tokenizer.eos_token_id: break
                
                print(tokenizer.decode([next_token_id]), end="", flush=True)
                generated = torch.cat([generated, next_token_tensor], dim=1)

        # 5. 최종 시퀀스에 대한 토큰별 사고 깊이 분석
        with torch.no_grad():
            _ = model(generated)
            ars = model.last_steps
            steps = model.last_ponder_steps[0]  # (seq_len,)
            
            print(f"\n\n[Reasoning Stats] Avg Steps: {ars:.2f}")
            print("[Token-wise Depth]:")
            for idx, token_id in enumerate(generated[0]):
                word = tokenizer.decode([token_id])
                depth = int(steps[idx].item())
                # 가독성을 위해 토큰[깊이] 형태로 출력
                print(f"{word}[{depth}]", end=" ", flush=True)
            print("\n" + "-" * 50)

if __name__ == "__main__":
    generate()
