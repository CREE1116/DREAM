import torch
import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def clean_text(text):
    text = re.sub(r'[\s\t\r\n]+', ' ', text)
    return text.strip()

class PackedDataset(Dataset):
    def __init__(self, tokenizer_path, dataset_names=["HAERAE-HUB/KOREAN-WEBTEXT", "HAERAE-HUB/KOREAN-SyntheticText-1.5B"], max_len=128):
        print(f"Initializing Packed Dataset with: {tokenizer_path}")
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = "<|pad|>"
        if self.tokenizer.eos_token is None: self.tokenizer.eos_token = "<|endoftext|>"
        
        cache_dir = "./data/cache"
        os.makedirs(cache_dir, exist_ok=True)
        # 여러 데이터셋의 이름을 합쳐서 캐시 파일명 생성
        ds_hash = "_".join([n.split("/")[-1] for n in dataset_names])
        bin_path = os.path.join(cache_dir, f"packed_{ds_hash}_len{max_len}.bin")
        
        if not os.path.exists(bin_path):
            print(f"[*] Cache not found. Tokenizing and Packing from {dataset_names}...")
            all_tokens_list = []
            
            for d_name in dataset_names:
                print(f"[*] Processing {d_name}...")
                ds = load_dataset(d_name, split="train", cache_dir="./data")
                
                for item in tqdm(ds, desc=f"Packing {d_name}"):
                    text = item.get("text", item.get("content", ""))
                    text = clean_text(text)
                    if not text or len(text.strip()) < 5: continue
                    
                    tokens = self.tokenizer.encode(text + self.tokenizer.eos_token, add_special_tokens=True)
                    all_tokens_list.extend(tokens)
            
            all_tokens_np = np.array(all_tokens_list, dtype=np.uint32)
            all_tokens_np.tofile(bin_path)
            del all_tokens_list, all_tokens_np
            print(f"[*] Saved binary file to: {bin_path}")

        self.all_tokens = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.max_len = max_len
        self.num_samples = (len(self.all_tokens) - 1) // max_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.max_len
        end = start + self.max_len + 1
        chunk = torch.from_numpy(self.all_tokens[start:end].astype(np.int64))
        return {
            "input_ids": chunk[:-1],
            "target_ids": chunk[1:]
        }

def get_dataloader(tokenizer_path, batch_size=16, max_len=128, dataset_names=["HAERAE-HUB/KOREAN-WEBTEXT", "HAERAE-HUB/KOREAN-SyntheticText-1.5B"], val_ratio=0.01):
    dataset = PackedDataset(tokenizer_path, dataset_names=dataset_names, max_len=max_len)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader
