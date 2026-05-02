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

def generate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer("checkpoints/dream_latest.pt", "checkpoints/tokenizer.json", device)

    print("\nDreamV2 Reasoning Engine — Active")
    while True:
        prompt = input("\nUser: ").strip()
        if prompt.lower() in ("exit", "q", "quit"): break
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids
        
        print("\nDREAM: ", end="", flush=True)
        with torch.no_grad():
            for i in range(128):
                logits = model(generated)
                next_token_id = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, dim=-1), 1).item()
                if next_token_id == tokenizer.eos_token_id: break
                
                print(tokenizer.decode([next_token_id]), end=" ", flush=True)
                generated = torch.cat([generated, torch.tensor([[next_token_id]], device=device)], dim=1)
        print()

if __name__ == "__main__":
    generate()
