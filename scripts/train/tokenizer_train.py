import os
import sys
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers import normalizers, decoders
from datasets import load_from_disk
from tqdm import tqdm

def train_tokenizer(vocab_size=16000):
    dataset_names = ["HAERAE-HUB/KOREAN-WEBTEXT", "HAERAE-HUB/KOREAN-SyntheticText-1.5B"]
    print(f"--- DreamV2 Tokenizer Training (Kiwi + BPE) ---")
    print(f"[*] Loading datasets: {dataset_names}")
    
    
    from datasets import load_dataset, concatenate_datasets
    # 전체 데이터셋을 로컬 ./data 폴더에 다운로드하여 캐싱
    datasets = [load_dataset(name, split="train", streaming=False, cache_dir="./data") for name in dataset_names]

    def batch_iterator(batch_size=1000):
        batch = []
        total_len = sum(len(ds) for ds in datasets)
        pbar = tqdm(total=total_len, desc="[*] Processing All Text")
        
        for dataset in datasets:
            for item in dataset:
                # 데이터셋마다 필드명이 다를 수 있으므로 확인 (일반적으로 'text' 또는 'content')
                text = item.get("text", item.get("content", ""))
                if not text: continue
                batch.append(text)
                pbar.update(1)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        if batch:
            yield batch
        pbar.close()

    # 2. Tokenizer 설정
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
        normalizers.Lowercase(),
        normalizers.Replace(r"[\x00-\x1f\x7f-\x9f]", ""),
        normalizers.Replace(r"\s+", " "),
    ])
    
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Trainer 설정
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[
            "<|endoftext|>",
            "<|startoftext|>",
            "<|pad|>",
            "[UNK]",
            "<sys>",
            "<usr>",
            "<bot>",
            "<|bot|>",
            "<|user|>",
        ]
    )

    # 4. 훈련 시작
    print(f"[*] Training Tokenizer (Vocab size: {vocab_size})...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 5. 후처리
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    
    # 6. 저장
    os.makedirs("checkpoints", exist_ok=True)
    tokenizer_path = "checkpoints/tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"\n[v] Success! Tokenizer saved to {tokenizer_path}")

if __name__ == "__main__":
    train_tokenizer()
