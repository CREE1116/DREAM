import os
import sys
import torch
import torch.nn.functional as F
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import DREAM
from transformers import PreTrainedTokenizerFast

def chat():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # SFT 학습 결과물 혹은 최신 체크포인트 경로
    checkpoint_path = "checkpoints/dream_sft_final.pt" 
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/dream_latest.pt"
        
    tokenizer_path = "checkpoints/tokenizer.json"

    if not os.path.exists(tokenizer_path):
        print(f"[Error] Tokenizer not found at {tokenizer_path}")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|pad|>"
    tokenizer.eos_token = "<|endoftext|>"

    # 모델 초기화 (SFT 포맷에 맞는 설정)
    model = DREAM(vocab_size=len(tokenizer), d_model=512, n_heads=8).to(device)
    
    if os.path.exists(checkpoint_path):
        print(f"[*] Loading model from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"[!] No checkpoint found. Inference with random weights.")
    
    model.eval()

    system_msg = "당신은 유능하고 친절하며 정직한 AI 어시스턴트 DreamV2입니다."
    
    print("\n" + "="*50)
    print("   DreamV2 SFT Chat Interface (Reasoning Engine)")
    print("="*50)
    print(" (Type 'exit' to quit)\n")

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except KeyboardInterrupt:
            break
            
        if not user_input: continue
        if user_input.lower() in ("exit", "q", "quit"):
            break
        
        # SFT 포맷 구성: <sys>\n...\n<usr>\n...\n<bot>\n
        prompt = f"<sys>\n{system_msg}\n<usr>\n{user_input}\n<bot>\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        print("DREAM > ", end="", flush=True)
        
        generated = input_ids
        with torch.no_grad():
            # 최대 256 토큰 생성
            for _ in range(256):
                logits = model(generated)
                # 샘플링 전략 (temperature 0.7)
                probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
                word = tokenizer.decode(next_token[0])
                print(word, end="", flush=True)
                
                generated = torch.cat([generated, next_token], dim=1)
        print("\n" + "-"*50)

if __name__ == "__main__":
    chat()
