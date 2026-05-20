import os
import sys
import torch
import torch.nn.functional as F
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import DREAM
from transformers import PreTrainedTokenizerFast

def sample_next_token(logits, input_ids, temperature=0.7, top_k=50, top_p=0.9, rep_penalty=1.2):
    logits = logits[0, -1, :].clone()
    
    # 1. Repetition Penalty
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
        
    # 4. Top-P
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
        
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).unsqueeze(0)

def chat():
    # === Inference Configuration ===
    TAU = 0.99           # Convergence threshold
    MIN_STEPS = 8        # Minimum reasoning steps (SFT에서는 조금 더 깊게 설정 권장)
    MAX_GEN_LEN = 256    # Maximum response length
    
    # Sampling Parameters
    TEMP = 0.7
    TOP_K = 50
    TOP_P = 0.9
    REP_PENALTY = 1.2
    # ===============================

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # SFT 학습 결과물 혹은 최신 체크포인트 경로
    checkpoint_path = "checkpoints/dream_sft_step_1000.pt" 
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/dream_latest.pt"
        
    tokenizer_path = "checkpoints/tokenizer.json"

    if not os.path.exists(tokenizer_path):
        print(f"[Error] Tokenizer not found at {tokenizer_path}")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|pad|>"
    tokenizer.eos_token = "<|endoftext|>"

    # 모델 초기화
    model = DREAM(vocab_size=len(tokenizer), d_model=512, n_heads=8).to(device)
    
    # 모델 파라미터 적용
    model.tau = TAU
    model.min_steps = MIN_STEPS

    if os.path.exists(checkpoint_path):
        print(f"[*] Loading model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    
    model.eval()

    system_msg = "당신은 유능하고 친절하며 정직한 AI 어시스턴트 DreamV2입니다."
    
    print("\n" + "="*50)
    print(f"   DreamV2 SFT Chat Interface (tau={TAU}, min_steps={MIN_STEPS})")
    print("="*50)
    print(" (Type 'exit' to quit)\n")

    while True:
        try:
            user_input = input(f"\nUser > ").strip()
        except KeyboardInterrupt:
            break
            
        if not user_input: continue
        if user_input.lower() in ("exit", "q", "quit"):
            break
        
        # SFT 포맷 구성
        prompt = f"<sys>\n{system_msg}\n<usr>\n{user_input}\n<bot>\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print("DREAM > ", end="", flush=True)
        
        generated = input_ids
        with torch.no_grad():
            for _ in range(MAX_GEN_LEN):
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
                
                if next_token_id == tokenizer.eos_token_id:
                    break
                    
                word = tokenizer.decode([next_token_id])
                print(word, end="", flush=True)
                
                generated = torch.cat([generated, next_token_tensor], dim=1)
        
        # 최종 시퀀스에 대한 사고 깊이 분석 추가
        print("\n\n" + "-"*15 + " Reasoning Analysis " + "-"*15)
        with torch.no_grad():
            _ = model(generated)
            steps = model.last_ponder_steps[0]
            for idx, token_id in enumerate(generated[0]):
                word = tokenizer.decode([token_id])
                depth = int(steps[idx].item())
                # 특수 토큰이나 줄바꿈 처리
                word = word.replace("\n", "\\n")
                print(f"{word}[{depth}]", end=" ", flush=True)
        print("\n" + "="*50)

if __name__ == "__main__":
    chat()
