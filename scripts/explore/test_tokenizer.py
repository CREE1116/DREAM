import os
import sys
# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from transformers import PreTrainedTokenizerFast

def test_tokenizer():
    tokenizer_path = "checkpoints/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"[Error] Tokenizer not found at {tokenizer_path}")
        return

    print(f"[*] Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    
    # 특수 토큰 설정
    tokenizer.pad_token = "<|pad|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.unk_token = "[UNK]"

    print(f"\n--- DreamV2 Tokenizer Interactive Test (Vocab: {len(tokenizer):,}) ---")
    print(" (Type 'exit' or 'q' to stop)\n")
    
    while True:
        try:
            text = input("Input > ").strip()
        except KeyboardInterrupt:
            print("\nExit.")
            break
            
        if not text: continue
        if text.lower() in ("exit", "q", "quit"):
            print("Exit.")
            break
        
        # 인코딩
        encoded = tokenizer.encode(text, add_special_tokens=False)
        byte_tokens = tokenizer.convert_ids_to_tokens(encoded)
        # 각 ID를 개별적으로 디코딩하여 사람이 읽기 편한 형태로 변환
        readable_tokens = [tokenizer.decode([id]) for id in encoded]
        
        # 시각화
        print(f"  Byte Tokens : {' | '.join(byte_tokens)}")
        print(f"  Clean Tokens: {' | '.join(readable_tokens)}")
        print(f"  IDs         : {encoded}")
        
        # 디코딩
        decoded = tokenizer.decode(encoded)
        print(f"  Decoded     : {decoded}")
        print("-" * 50)

if __name__ == "__main__":
    test_tokenizer()
