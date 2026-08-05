import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
import random
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import requests
import logging
import json
from pathlib import Path
import hashlib
import time
import pandas as pd
from tqdm import tqdm
import threading
import queue
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Frame, Label

# ==========================================
# 1. 로깅 및 환경 설정
# ==========================================
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "training.log", encoding='utf-8')]
)
logger = logging.getLogger(__name__)

LOSS_HISTORY_FILE = LOG_DIR / "loss_history.json"
loss_history = []

def save_loss_history():
    try:
        with open(LOSS_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(loss_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Loss history save failed: {e}")

def load_loss_history():
    global loss_history
    if LOSS_HISTORY_FILE.exists():
        try:
            with open(LOSS_HISTORY_FILE, 'r', encoding='utf-8') as f:
                loss_history = json.load(f)
        except Exception:
            loss_history = []

# ==========================================
# 2. 데이터셋 관리
# ==========================================
DATASETS_DIR = Path("./datasets")
DATASETS_CACHE_DIR = DATASETS_DIR / "cache"
DATASETS_MANIFEST_FILE = DATASETS_DIR / "datasets_manifest.json"

class DatasetManager:
    DATASETS_CONFIG = [
        {"name": "maywell/korean_textbooks", "config": "tiny-textbooks", "split": "train", "text_key": "text"},
        {"name": "squarelike/OpenOrca-gugugo-ko", "config": None, "split": "train", "text_keys": ["question", "response"], "system_key": "system_prompt"},
        {"name": "beomi/KoAlpaca-v1.1a", "config": None, "split": "train", "text_keys": ["instruction", "output"]}
    ]
    
    def __init__(self):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if DATASETS_MANIFEST_FILE.exists():
            return json.load(open(DATASETS_MANIFEST_FILE, 'r', encoding='utf-8'))
        return {}

    def download_dataset(self, config):
        h = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        if h in self.manifest: return self.manifest[h]['path']
        
        logger.info(f"📥 Downloading {config['name']}...")
        local_path = DATASETS_CACHE_DIR / h
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            ds = load_dataset(config['name'], name=config.get('config'), split=config['split'], cache_dir=str(DATASETS_CACHE_DIR))
            p = local_path / "data.parquet"
            ds.to_parquet(str(p))
            self.manifest[h] = {'path': str(local_path)}
            json.dump(self.manifest, open(DATASETS_MANIFEST_FILE, 'w'), indent=2)
            return str(local_path)
        except Exception as e:
            logger.error(f"Download failed: {e}"); return None

    def get_all(self):
        return [self.download_dataset(c) for c in self.DATASETS_CONFIG if self.download_dataset(c)]

class LocalKoreanDataset(Dataset):
    def __init__(self, paths, tokenizer, max_len=256, samples_per_ds=None):
        self.tokenizer, self.max_len, self.samples = tokenizer, max_len, []
        for p in paths:
            ds = HFDataset.from_parquet(str(Path(p)/"data.parquet"))
            if samples_per_ds: ds = ds.select(range(min(len(ds), samples_per_ds)))
            for item in ds:
                if "text" in item: txt = item["text"]
                elif "question" in item: txt = f"### 지시: {item.get('question')}\n### 응답: {item.get('response')}"
                elif "instruction" in item: txt = f"### 지시: {item['instruction']}\n### 응답: {item['output']}"
                else: continue
                if len(txt) > 5: self.samples.append(txt)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.samples[idx] + self.tokenizer.eos_token, truncation=True, max_length=self.max_len)
        encoded += [self.tokenizer.pad_token_id] * (self.max_len - len(encoded))
        return torch.tensor(encoded[:self.max_len], dtype=torch.long)

# ==========================================
# 3. 모델 아키텍처 (1B급 설정)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(head_dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(dim, hidden_dim, bias=False), nn.Linear(hidden_dim, dim, bias=False), nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.wq, self.wk, self.wv, self.wo = [nn.Linear(dim, dim, bias=False) for _ in range(3)] + [nn.Linear(dim, dim, bias=False)]
    def forward(self, x, f_cos, f_sin, kv_cache=None):
        b, s, d = x.shape
        q, k, v = [l(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2) for l in [self.wq, self.wk, self.wv]]
        q, k = apply_rotary_emb(q, f_cos, f_sin), apply_rotary_emb(k, f_cos, f_sin)
        if kv_cache:
            pk, pv = kv_cache
            k, v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True if s > 1 else False)
        return self.wo(out.transpose(1, 2).contiguous().view(b, s, d)), (k.detach(), v.detach())

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.attention, self.feed_forward = Attention(dim, n_heads), SwiGLU(dim, hidden_dim)
        self.attention_norm, self.ffn_norm = RMSNorm(dim), RMSNorm(dim)
    def forward(self, x, f_cos, f_sin, kv_cache=None):
        h, new_kv = self.attention(self.attention_norm(x), f_cos, f_sin, kv_cache)
        x = x + h
        return x + self.feed_forward(self.ffn_norm(x)), new_kv

class KoreanLLM(nn.Module):
    def __init__(self, vocab_size, pad_token_id, dim=1920, n_layers=20, n_heads=10, max_seq_len=2048):
        super().__init__()
        self.vocab_size, self.pad_token_id, self.dim, self.n_layers = vocab_size, pad_token_id, dim, n_layers
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, int(dim * 2.5)) for _ in range(n_layers)])
        self.norm, self.output = RMSNorm(dim), nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight
        f_cos, f_sin = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos), self.register_buffer("f_sin", f_sin)

    def forward(self, tokens, labels=None, kv_caches=None):
        b, s = tokens.shape
        x = self.embed(tokens)
        start_pos = kv_caches[0][0].shape[2] if kv_caches and kv_caches[0][0] is not None else 0
        f_cos, f_sin = self.f_cos[start_pos:start_pos+s], self.f_sin[start_pos:start_pos+s]
        new_kvs = []
        for i, layer in enumerate(self.layers):
            if self.training: x, kv = checkpoint(layer, x, f_cos, f_sin, None, use_reentrant=False)
            else: x, kv = layer(x, f_cos, f_sin, kv_caches[i] if kv_caches else None)
            new_kvs.append(kv)
        logits = self.output(self.norm(x))
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1), ignore_index=self.pad_token_id) if labels is not None else None
        return logits, loss, new_kvs

# ==========================================
# 4. 생성 및 GUI 모니터링
# ==========================================
@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=80, temperature=0.7, top_p=0.9, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(f"### 지시: {prompt}\n### 응답:", return_tensors="pt").to(device)
    kv_caches, output_tokens = None, tokens
    for _ in range(max_tokens):
        input_t = output_tokens[:, -1:] if kv_caches else output_tokens
        logits, _, kv_caches = model(input_t, kv_caches=kv_caches)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        # Nucleus Sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumsum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0.0
        next_token = torch.multinomial(probs / probs.sum(), 1)
        output_tokens = torch.cat([output_tokens, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id: break
    res = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return res.split("### 응답:")[-1].strip()

class TrainingMonitorGUI:
    def __init__(self, tokenizer, model_config, train_model):
        self.tokenizer, self.model_config, self.train_model = tokenizer, model_config, train_model
        self.chat_model = None # CPU 추론용
        self.msg_queue = queue.Queue()
        self.root = tk.Tk()
        self.root.title("KoreanLLM Monitor (CPU Inference)")
        self.root.geometry("1000x600")
        
        # UI 레이아웃
        f_left = Frame(self.root); f_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(5, 4)); self.line, = self.ax.plot([], [], 'b-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=f_left); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        f_right = Frame(self.root); f_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.chat_area = scrolledtext.ScrolledText(f_right, height=20, state='disabled')
        self.chat_area.pack(fill=tk.BOTH, expand=True)
        self.input_box = Entry(f_right); self.input_box.pack(fill=tk.X); self.input_box.bind("<Return>", self.send)
        Button(f_right, text="학습 상태 동기화 (CPU로 복사)", command=self.sync_model).pack()
        
        self.root.after(1000, self.update_gui)

    def sync_model(self):
        """현재 학습 중인 GPU 모델의 가중치를 CPU 모델로 복사"""
        self.log_chat("[시스템] 모델 가중치 복사 중 (CPU)...")
        if self.chat_model is None:
            self.chat_model = KoreanLLM(**self.model_config).to('cpu')
        # 학습 모델의 state_dict를 CPU로 복사하여 로드
        state_dict = {k: v.cpu() for k, v in self.train_model.state_dict().items()}
        self.chat_model.load_state_dict(state_dict)
        self.chat_model.eval()
        self.log_chat("[시스템] 동기화 완료. 이제 CPU에서 안전하게 테스트 가능합니다.")

    def send(self, e=None):
        p = self.input_box.get(); self.input_box.delete(0, tk.END)
        if not self.chat_model: self.log_chat("[오류] 먼저 동기화 버튼을 눌러주세요."); return
        self.log_chat(f"나: {p}")
        threading.Thread(target=self._gen_thread, args=(p,), daemon=True).start()

    def _gen_thread(self, p):
        try:
            res = generate(self.chat_model, self.tokenizer, p, device='cpu')
            self.msg_queue.put({"type": "chat", "txt": f"모델: {res}"})
        except Exception as e: self.msg_queue.put({"type": "chat", "txt": f"[오류]: {e}"})

    def log_chat(self, t):
        self.chat_area.config(state='normal'); self.chat_area.insert(tk.END, t + "\n"); self.chat_area.see(tk.END); self.chat_area.config(state='disabled')

    def update_gui(self):
        try:
            while True:
                m = self.msg_queue.get_nowait()
                if m["type"] == "loss":
                    self.line.set_data([h["step"] for h in loss_history], [h["loss"] for h in loss_history])
                    self.ax.relim(); self.ax.autoscale_view(); self.canvas.draw()
                elif m["type"] == "chat": self.log_chat(m["txt"])
        except queue.Empty: pass
        self.root.after(1000, self.update_gui)

# ==========================================
# 5. 메인 학습 루프 (휴식 로직 포함)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
    tokenizer.pad_token = tokenizer.eos_token
    
    manager = DatasetManager()
    ds = LocalKoreanDataset(manager.get_all(), tokenizer)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    
    model_config = dict(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, dim=1920, n_layers=20, n_heads=10, max_seq_len=256)
    model = KoreanLLM(**model_config).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 200, 50000)
    scaler = torch.amp.GradScaler('cuda')
    
    # GUI 실행
    gui = TrainingMonitorGUI(tokenizer, model_config, model)
    threading.Thread(target=gui.root.mainloop, daemon=True).start()
    
    step = 0
    running_loss = 0.0
    accum_steps = 32
    
    logger.info("🚀 학습 시작...")
    model.train()
    
    try:
        while True:
            for batch in loader:
                batch = batch.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss, _ = model(batch, labels=batch)
                    loss_scaled = loss / accum_steps
                
                scaler.scale(loss_scaled).backward()
                running_loss += loss.item()
                
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    actual_step = (step + 1) // accum_steps
                    avg_loss = running_loss / accum_steps
                    loss_history.append({"step": actual_step, "loss": avg_loss})
                    gui.msg_queue.put({"type": "loss"})
                    
                    logger.info(f"Step {actual_step} | Loss {avg_loss:.4f}")
                    running_loss = 0.0
                    
                    # --------------------------------------------------
                    # [추가] 250 스텝마다 5초간 휴식 (VRAM 발열 및 시스템 안정화)
                    # --------------------------------------------------
                    if actual_step % 250 == 0:
                        logger.info("⏸️ 250스텝 도달: 5초간 휴식합니다...")
                        time.sleep(5)
                    
                    # 체크포인트 저장
                    if actual_step % 1000 == 0:
                        torch.save({'model_state_dict': model.state_dict()}, f"checkpoints/llm_{actual_step}.pth")
                
                step += 1
    except KeyboardInterrupt:
        logger.info("학습 중단")

if __name__ == "__main__":
    main()