import os
import sys
import torch
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import DREAM

def check_parameters():
    print("--- DreamV2 Parameter Analysis ---")
    
    # 더미 파라미터로 모델 초기화 (분석용)
    vocab_size = 16000 
    model = DREAM(vocab_size=vocab_size, d_model=512, n_heads=8)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[*] Model Configuration: d_model=512, n_heads=8, vocab_size={vocab_size}")
    print(f"[*] Total Parameters: {total_params:,}")
    print(f"[*] Trainable Parameters: {trainable_params:,}")
    print("-" * 40)
    
    # 주요 컴포넌트별 파라미터 확인
    print("[Layer-wise breakdown]")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:30} | {param.numel():10,}")

if __name__ == "__main__":
    check_parameters()
