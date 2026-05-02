import os
import sys
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import DREAM
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import os
from tqdm import tqdm

class SFTDataset(Dataset):
    def __init__(self, tokenizer_path, dataset_name="heegyu/open-korean-instructions", max_len=512):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        # 특수 토큰 확인 및 추가 설정
        special_tokens = ["<sys>", "<usr>", "<bot>", "<|pad|>", "<|endoftext|>"]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                print(f"[Warning] Token {token} not found in tokenizer. Please re-train tokenizer if possible.")
        
        self.tokenizer.pad_token = "<|pad|>"
        self.tokenizer.eos_token = "<|endoftext|>"
        
        print(f"[*] Loading SFT dataset: {dataset_name}...")
        try:
            ds = load_dataset(dataset_name, split="train", cache_dir="./data")
        except Exception as e:
            print(f"[!] HF loading failed: {e}")
            ds = []

        self.samples = []
        for item in tqdm(ds, desc="Processing SFT Data"):
            text = item.get("text", "")
            if not text or "<bot>" not in text: continue
            self.samples.append(text)
        
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        if not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token
            
        enc = self.tokenizer.encode(text, truncation=True, max_length=self.max_len, add_special_tokens=False)
        
        input_ids = torch.tensor(enc[:-1], dtype=torch.long)
        target_ids = torch.tensor(enc[1:], dtype=torch.long)
        
        # 레이블 마스킹 로직: <bot> 이전의 모든 토큰은 -100으로 설정 (Loss 계산 제외)
        # <bot> 토큰의 ID를 찾음
        bot_token_id = self.tokenizer.convert_tokens_to_ids("<bot>")
        
        # target_ids에서 <bot> 토큰이 나타나는 위치를 찾음
        labels = target_ids.clone()
        mask_until = -1
        for i, tid in enumerate(labels):
            if tid == bot_token_id:
                mask_until = i
                break
        
        if mask_until != -1:
            labels[:mask_until+1] = -100 # <bot> 토큰 포함 그 이전까지 마스킹
            
        # 패딩 처리 (max_len에 맞춤)
        pad_len = (self.max_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            
        return {"input_ids": input_ids, "target_ids": labels}

def train_sft():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_path = "checkpoints/tokenizer.json"
    checkpoint_path = "checkpoints/dream_latest.pt"

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    model, _ = DREAM(vocab_size=len(tokenizer)).to(device), None
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])

    dataset = SFTDataset(tokenizer_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(3):
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.reshape(-1, len(tokenizer)), target_ids.reshape(-1))
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    torch.save(model.state_dict(), "checkpoints/dream_sft_final.pt")

if __name__ == "__main__":
    train_sft()
