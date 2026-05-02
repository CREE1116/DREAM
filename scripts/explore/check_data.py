import os
import glob
import numpy as np

def check_packed_data():
    cache_dir = "data/cache"
    # 재구성된 구조에서도 루트 기준으로 작동하도록 경로 설정
    bin_files = glob.glob(os.path.join(cache_dir, "*.bin"))
    
    if not bin_files:
        print(f"[!] No packed binary files found in {cache_dir}")
        print("[*] Run training first to generate packed data.")
        return

    print(f"--- DreamV2 Data Analysis ---")
    for bin_path in bin_files:
        file_size = os.path.getsize(bin_path)
        # uint32 = 4 bytes per token
        total_tokens = file_size // 4
        
        print(f"\n[File]: {os.path.basename(bin_path)}")
        print(f"[*] File Size    : {file_size / (1024**2):.2f} MB")
        print(f"[*] Total Tokens : {total_tokens:,} tokens")
        
        # 시퀀스 길이 128, 256, 512 기준 샘플 수
        for s_len in [128, 256, 512]:
            num_samples = total_tokens // s_len
            print(f"[*] Samples ({s_len:3}) : {num_samples:,}")

if __name__ == "__main__":
    check_packed_data()
