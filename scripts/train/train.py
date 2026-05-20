import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# pyrefly: ignore [missing-import]
from src.model import DREAM
# pyrefly: ignore [missing-import]
from src.data_loader import get_dataloader
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from transformers import PreTrainedTokenizerFast

def train():
    log_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/dreamv2_{log_id}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_path = "checkpoints/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        print(f"[Error] Tokenizer not found at {tokenizer_path}. Run tokenizer_train.py first.")
        return
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.pad_token = "<|pad|>"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.bos_token = "<|startoftext|>"
    
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    
    model = DREAM(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        max_steps=64,
        tau=0.999,
        min_steps=1,
        freeze_dropout=0.5
    ).to(device)

    train_loader, val_loader = get_dataloader(
        tokenizer_path, 
        batch_size=16, 
        max_len=128, 
        dataset_names=["HAERAE-HUB/KOREAN-WEBTEXT", "HAERAE-HUB/KOREAN-SyntheticText-1.5B"]
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    num_epochs = 10
    total_steps = num_epochs * len(train_loader)
    warmup_steps = 2000
    
    scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    global_step = 0
    start_epoch = 0
    latest_path = "checkpoints/dream_latest.pt"
    
    if os.path.exists(latest_path):
        print(f"[*] Resuming from {latest_path}...")
        ckpt = torch.load(latest_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            # Fallback if the whole file is just the state dict (some older saves might do this)
            model.load_state_dict(ckpt)
            
        # Load optimizer state
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("[Warning] 'optimizer_state_dict' not found in checkpoint. Skipping.")

        # Load scheduler state
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            print("[Warning] 'scheduler_state_dict' not found in checkpoint. Skipping.")

        global_step = ckpt.get('global_step', 0)
        start_epoch = ckpt.get('epoch', 0)

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                global_step += 1
                input_seq = batch["input_ids"].to(device)
                target_seq = batch["target_ids"].to(device)
                
                optimizer.zero_grad(set_to_none=True)
                output = model(input_seq, is_causal=True)
                loss = criterion(output.reshape(-1, vocab_size), target_seq.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                if global_step % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'pnd': f"{model.last_steps:.1f}",
                        'fz%': f"{model.last_freeze_ratio*100:.1f}%"
                    })
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                    writer.add_scalar("Reasoning/ponder_steps", model.last_steps, global_step)
                    writer.add_scalar("Efficiency/freeze_ratio", model.last_freeze_ratio, global_step)

                if global_step % 500 == 0:
                    # Validation Loop
                    model.eval()
                    with torch.no_grad():
                        val_losses = []
                        for i, val_batch in enumerate(val_loader):
                            if i >= 50: break
                            out = model(val_batch["input_ids"].to(device), is_causal=True)
                            val_loss = criterion(out.reshape(-1, vocab_size), 
                                                val_batch["target_ids"].to(device).reshape(-1))
                            val_losses.append(val_loss.item())
                        if len(val_losses) > 0:
                            writer.add_scalar("Loss/val", sum(val_losses)/len(val_losses), global_step)
                    model.train()

                    # Checkpoint Save
                    save_data = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch,
                        'config': {'vocab_size': vocab_size, 'd_model': 512, 'n_heads': 8}
                    }
                    torch.save(save_data, f"checkpoints/dream_step_{global_step}.pt")
                    torch.save(save_data, latest_path)

                if device.type == 'mps' and global_step % 50 == 0:
                    torch.mps.empty_cache()

    except KeyboardInterrupt:
        print(f"\nSaving state...")
        save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'config': {'vocab_size': vocab_size, 'd_model': 512, 'n_heads': 8}
        }
        torch.save(save_data, latest_path)

    writer.close()

if __name__ == "__main__":
    train()
