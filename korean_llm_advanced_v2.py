import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, Dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset as HFDataset
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
matplotlib.use('TkAgg')  # GUI 백엔드
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import scrolledtext, Entry, Button, Frame, Label, StringVar
import copy
import traceback

# ==========================================
# 로깅 설정 (파일 저장 추가)
# ==========================================
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "training.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Loss 히스토리 저장용
LOSS_HISTORY_FILE = LOG_DIR / "loss_history.json"
loss_history = []  # [{"step": int, "loss": float, "lr": float, "time": str}, ...]

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
            logger.info(f"✅ Loaded {len(loss_history)} previous loss records")
        except Exception as e:
            logger.warning(f"Failed to load loss history: {e}")
            loss_history = []

# ==========================================
# 데이터셋 기본 설정
# ==========================================

DATASETS_DIR = Path("./datasets")
DATASETS_CACHE_DIR = DATASETS_DIR / "cache"
DATASETS_MANIFEST_FILE = DATASETS_DIR / "datasets_manifest.json"

def ensure_datasets_dir():
    """데이터셋 디렉토리 생성"""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Datasets directory ready: {DATASETS_DIR.absolute()}")

# ==========================================
# 데이터셋 다운로드 및 관리
# ==========================================

class DatasetManager:
    """로컬 데이터셋 관리 클래스"""
    
    DATASETS_CONFIG = [
        {
            "name": "maywell/korean_textbooks",
            "config": "tiny-textbooks",
            "split": "train",
            "text_key": "text"
        },
        {
            "name": "squarelike/OpenOrca-gugugo-ko",
            "config": None,
            "split": "train",
            "text_keys": ["question", "response"]
        },
        {
            "name": "beomi/KoAlpaca-v1.1a",
            "config": None,
            "split": "train",
            "text_keys": ["instruction", "output"]
        }
    ]
    
    def __init__(self, cache_dir: Path = DATASETS_CACHE_DIR):
        self.cache_dir = cache_dir
        self.manifest = self._load_manifest()
        ensure_datasets_dir()
    
    def _load_manifest(self) -> Dict:
        """매니페스트 파일 로드"""
        if DATASETS_MANIFEST_FILE.exists():
            with open(DATASETS_MANIFEST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_manifest(self):
        """매니페스트 파일 저장"""
        with open(DATASETS_MANIFEST_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
    
    def _get_dataset_hash(self, config: Dict) -> str:
        """데이터셋 config의 해시값 생성"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def download_dataset(self, config: Dict, force: bool = False) -> Optional[str]:
        """
        데이터셋 다운로드
        - OpenOrca-gugugo-ko는 parquet / streaming 청크 우선
        - 실패 시 자세한 에러 로그 출력
        """
        dataset_hash = self._get_dataset_hash(config)
        dataset_name = config['name']
        
        # 캐시 확인
        if dataset_hash in self.manifest and not force:
            cached_path = self.manifest[dataset_hash].get('path')
            if cached_path and Path(cached_path).exists():
                logger.info(f"✅ Using cached dataset: {dataset_name}")
                return cached_path
        
        logger.info(f"📥 Downloading {dataset_name}...")
        
        last_error = None
        error_details = []
        
        # ============================================================
        # OpenOrca-gugugo-ko 전용 처리 (JSON OverflowError 회피)
        # ============================================================
        if "OpenOrca-gugugo-ko" in dataset_name:
            strategies = [
                {"name": "parquet_auto", "desc": "HF auto-converted parquet"},
                {"name": "parquet_manual", "desc": "수동 parquet 경로 지정"},
                {"name": "streaming_chunk", "desc": "streaming + 청크 저장"},
            ]
            
            for attempt, strategy in enumerate(strategies, 1):
                try:
                    logger.info(f"🔄 Attempt {attempt}/{len(strategies)} - Strategy: {strategy['name']} ({strategy['desc']})")
                    
                    if strategy["name"] == "parquet_auto":
                        ds = load_dataset(
                            config["name"],
                            split=config["split"],
                            cache_dir=str(self.cache_dir),
                            verification_mode="no_checks",
                        )
                    
                    elif strategy["name"] == "parquet_manual":
                        ds = load_dataset(
                            "parquet",
                            data_files={
                                "train": "hf://datasets/squarelike/OpenOrca-gugugo-ko@~parquet/default/train/*.parquet"
                            },
                            split="train",
                            cache_dir=str(self.cache_dir),
                        )
                    
                    elif strategy["name"] == "streaming_chunk":
                        logger.info("📥 Streaming으로 로드 후 청크 단위 저장 중...")
                        stream_ds = load_dataset(
                            config["name"],
                            split=config["split"],
                            streaming=True,
                            cache_dir=str(self.cache_dir),
                        )
                        
                        local_path = self.cache_dir / f"{dataset_hash}"
                        local_path.mkdir(exist_ok=True, parents=True)
                        
                        chunk_size = 20000
                        buffer = []
                        total = 0
                        chunk_idx = 0
                        parquet_files = []
                        
                        for item in stream_ds:
                            buffer.append(item)
                            if len(buffer) >= chunk_size:
                                chunk_ds = HFDataset.from_list(buffer)
                                chunk_file = local_path / f"chunk_{chunk_idx:04d}.parquet"
                                chunk_ds.to_parquet(str(chunk_file))
                                parquet_files.append(str(chunk_file))
                                total += len(buffer)
                                logger.info(f"   ↳ saved chunk {chunk_idx} ({total} examples so far)")
                                buffer = []
                                chunk_idx += 1
                        
                        # 남은 버퍼 저장
                        if buffer:
                            chunk_ds = HFDataset.from_list(buffer)
                            chunk_file = local_path / f"chunk_{chunk_idx:04d}.parquet"
                            chunk_ds.to_parquet(str(chunk_file))
                            parquet_files.append(str(chunk_file))
                            total += len(buffer)
                        
                        # 청크들을 하나로 합치기
                        logger.info("📥 Merging chunks into single parquet...")
                        full_ds = load_dataset(
                            "parquet",
                            data_files=parquet_files,
                            split="train"
                        )
                        final_path = local_path / "data.parquet"
                        full_ds.to_parquet(str(final_path))
                        
                        # 임시 청크 파일 삭제
                        for f in parquet_files:
                            try:
                                Path(f).unlink()
                            except Exception:
                                pass
                        
                        self.manifest[dataset_hash] = {
                            'name': dataset_name,
                            'config': config,
                            'path': str(local_path),
                            'num_examples': total,
                            'download_strategy': strategy['name']
                        }
                        self._save_manifest()
                        
                        logger.info(f"✅ Dataset saved via streaming_chunk: {local_path} ({total} examples)")
                        return str(local_path)
                    
                    # parquet_auto / parquet_manual 성공 시 공통 저장
                    local_path = self.cache_dir / f"{dataset_hash}"
                    local_path.mkdir(exist_ok=True, parents=True)
                    
                    ds.to_parquet(str(local_path / "data.parquet"))
                    
                    self.manifest[dataset_hash] = {
                        'name': dataset_name,
                        'config': config,
                        'path': str(local_path),
                        'num_examples': len(ds),
                        'download_strategy': strategy['name']
                    }
                    self._save_manifest()
                    
                    logger.info(f"✅ Dataset saved: {local_path} ({len(ds)} examples) via {strategy['name']}")
                    return str(local_path)
                
                except Exception as e:
                    last_error = e
                    error_type = type(e).__name__
                    error_msg = str(e)
                    tb_str = traceback.format_exc()
                    
                    error_details.append({
                        "attempt": attempt,
                        "strategy": strategy['name'],
                        "error_type": error_type,
                        "error_msg": error_msg,
                        "traceback": tb_str
                    })
                    
                    logger.warning(f"⚠️ Attempt {attempt} failed ({strategy['name']})")
                    logger.warning(f"   ↳ Error Type : {error_type}")
                    logger.warning(f"   ↳ Error Msg  : {error_msg}")
                    logger.warning(f"   ↳ Full Traceback:\n{tb_str}")
                    
                    if attempt < len(strategies):
                        wait_time = attempt * 2
                        logger.info(f"⏳ Waiting {wait_time} seconds before next attempt...")
                        time.sleep(wait_time)
            
            # 모든 전략 실패
            logger.error("=" * 80)
            logger.error(f"❌ Failed to download {dataset_name} after {len(strategies)} strategies")
            logger.error("=" * 80)
            for detail in error_details:
                logger.error(f"[Attempt {detail['attempt']}] Strategy: {detail['strategy']}")
                logger.error(f"  - Type   : {detail['error_type']}")
                logger.error(f"  - Message: {detail['error_msg']}")
                logger.error(f"  - Traceback:\n{detail['traceback']}")
                logger.error("-" * 60)
            logger.error(f"📌 Last error summary: {type(last_error).__name__}: {last_error}")
            logger.error("=" * 80)
            return None
        
        # ============================================================
        # 일반 데이터셋 처리
        # ============================================================
        strategies = [
            {"name": "standard", "streaming": False, "force_redownload": False},
            {"name": "streaming", "streaming": True, "force_redownload": False},
            {"name": "force_redownload", "streaming": False, "force_redownload": True},
        ]
        
        for attempt, strategy in enumerate(strategies, 1):
            try:
                logger.info(f"🔄 Attempt {attempt}/3 - Strategy: {strategy['name']}")
                
                load_kwargs = {
                    "path": config["name"],
                    "split": config["split"],
                    "cache_dir": str(self.cache_dir),
                }
                
                if config.get("config"):
                    load_kwargs["name"] = config["config"]
                
                if strategy["streaming"]:
                    load_kwargs["streaming"] = True
                
                if strategy["force_redownload"]:
                    load_kwargs["download_mode"] = "force_redownload"
                
                ds = load_dataset(**load_kwargs)
                
                if strategy["streaming"]:
                    logger.info("📥 Converting streaming dataset to regular dataset...")
                    ds = HFDataset.from_list(list(ds))
                
                local_path = self.cache_dir / f"{dataset_hash}"
                local_path.mkdir(exist_ok=True)
                
                ds.to_parquet(str(local_path / "data.parquet"))
                
                self.manifest[dataset_hash] = {
                    'name': dataset_name,
                    'config': config,
                    'path': str(local_path),
                    'num_examples': len(ds),
                    'download_strategy': strategy['name']
                }
                self._save_manifest()
                
                logger.info(f"✅ Dataset saved: {local_path} ({len(ds)} examples) via {strategy['name']}")
                return str(local_path)
            
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                error_msg = str(e)
                tb_str = traceback.format_exc()
                
                error_details.append({
                    "attempt": attempt,
                    "strategy": strategy['name'],
                    "error_type": error_type,
                    "error_msg": error_msg,
                    "traceback": tb_str
                })
                
                logger.warning(f"⚠️ Attempt {attempt} failed ({strategy['name']})")
                logger.warning(f"   ↳ Error Type : {error_type}")
                logger.warning(f"   ↳ Error Msg  : {error_msg}")
                logger.warning(f"   ↳ Full Traceback:\n{tb_str}")
                
                if attempt < 3:
                    wait_time = attempt * 2
                    logger.info(f"⏳ Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)
        
        # 모든 시도 실패
        logger.error("=" * 80)
        logger.error(f"❌ Failed to download {dataset_name} after 3 attempts")
        logger.error("=" * 80)
        for detail in error_details:
            logger.error(f"[Attempt {detail['attempt']}] Strategy: {detail['strategy']}")
            logger.error(f"  - Type   : {detail['error_type']}")
            logger.error(f"  - Message: {detail['error_msg']}")
            logger.error(f"  - Traceback:\n{detail['traceback']}")
            logger.error("-" * 60)
        logger.error(f"📌 Last error summary: {type(last_error).__name__}: {last_error}")
        logger.error("=" * 80)
        return None
    
    def get_or_download_all(self, force: bool = False) -> List[str]:
        """모든 데이터셋 다운로드 또는 캐시 로드"""
        paths = []
        failed = []
        
        for config in self.DATASETS_CONFIG:
            path = self.download_dataset(config, force=force)
            if path:
                paths.append(path)
            else:
                failed.append(config['name'])
        
        logger.info(f"✅ Ready with {len(paths)} datasets")
        
        if failed:
            logger.warning("=" * 60)
            logger.warning(f"⚠️ 다음 데이터셋 다운로드 실패 ({len(failed)}개):")
            for name in failed:
                logger.warning(f"   - {name}")
            logger.warning("=" * 60)
        
        return paths


# ==========================================
# 로컬 데이터셋 클래스
# ==========================================

class LocalKoreanDataset(Dataset):
    """로컬 파일에서 로드하는 한국어 데이터셋"""
    
    def __init__(
        self,
        dataset_paths: List[str],
        tokenizer,
        max_len: int = 256,
        data_samples_per_dataset: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        logger.info("📚 Loading local datasets...")
        
        for dataset_path in dataset_paths:
            try:
                parquet_file = Path(dataset_path) / "data.parquet"
                if not parquet_file.exists():
                    logger.warning(f"Parquet file not found: {parquet_file}")
                    continue
                
                ds = HFDataset.from_parquet(str(parquet_file))
                
                if data_samples_per_dataset:
                    ds = ds.select(range(min(len(ds), data_samples_per_dataset)))
                
                texts = self._extract_texts(ds)
                self.samples.extend(texts)
                
                logger.info(f"✅ Loaded {len(texts)} samples from {Path(dataset_path).name}")
            
            except Exception as e:
                logger.error(f"❌ Error loading dataset from {dataset_path}: {e}")
                logger.error(f"   Full traceback:\n{traceback.format_exc()}")
                continue
        
        logger.info(f"✅ Total samples loaded: {len(self.samples)}")
    
    def _extract_texts(self, ds) -> List[str]:
        """데이터셋에서 텍스트 추출"""
        texts = []
        
        for item in ds:
            text = None
            
            if "text" in item and item["text"]:
                text = item["text"]
            elif "instruction" in item and "output" in item:
                text = f"### 지시: {item['instruction']}\n### 응답: {item['output']}"
            elif "question" in item and "response" in item:  # OpenOrca-gugugo-ko
                system = item.get("system_prompt", "")
                if system:
                    text = f"### 시스템: {system}\n### 질문: {item['question']}\n### 응답: {item['response']}"
                else:
                    text = f"### 질문: {item['question']}\n### 응답: {item['response']}"
            elif "question" in item and "answer" in item:
                text = f"### 질문: {item['question']}\n### 답변: {item['answer']}"
            elif "prompt" in item and "response" in item:
                text = f"### 프롬프트: {item['prompt']}\n### 응답: {item['response']}"
            
            if text and len(text) > 5:
                texts.append(text)
        
        return texts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.samples[idx]
        
        try:
            text_with_eos = text + self.tokenizer.eos_token
            
            encoded = self.tokenizer.encode(
                text_with_eos,
                truncation=True,
                max_length=self.max_len
            )
            
            if len(encoded) < self.max_len:
                encoded += [self.tokenizer.pad_token_id] * (self.max_len - len(encoded))
            else:
                encoded = encoded[:self.max_len]
            
            return torch.tensor(encoded, dtype=torch.long)
        
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return torch.full((self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
    """배치 콜레이션"""
    return torch.stack(batch)

# ==========================================
# 1. 아키텍처 (개선된 버전 - 기존 코드 유지)
# ==========================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# ✅ BUG FIX #1: RoPE - head_dim 기준으로 계산 (차원 일치)
def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPE(Rotary Position Embedding) 주파수 사전계산
    수정사항: dim 대신 head_dim 사용 (Llama: θ_j = 10000^(-2j/d_head))
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

# ✅ BUG FIX #2: apply_rotary_emb - 차원 처리 정확화
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE 적용 (정확한 차원 처리)"""
    seq, head_dim_2 = cos.shape  # head_dim_2 = head_dim // 2
    head_dim = head_dim_2 * 2
    
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    
    # 차원 맞추기: cos/sin을 (1, 1, seq, head_dim//2)로 방송
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    """SwiGLU 활성화 함수"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """Multi-Head Attention with KV-Cache"""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(
        self, 
        x: torch.Tensor, 
        f_cos: torch.Tensor, 
        f_sin: torch.Tensor, 
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, s, d = x.shape
        
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        
        # ✅ BUG FIX #2: RoPE 적용 (정확한 차원)
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        # ✅ KV Cache (training에선 None, inference에선 누적)
        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)  # seq 차원에서 누적
            v = torch.cat([pv, v], dim=2)
        
        new_kv = (k.detach(), v.detach())
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and s > 1)
        )
        
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv

class TransformerBlock(nn.Module):
    """Transformer 블록"""
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
    
    def forward(
        self,
        x: torch.Tensor,
        f_cos: torch.Tensor,
        f_sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv

class KoreanLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128256,
        pad_token_id: int = 128004,
        dim: int = 1920,
        n_layers: int = 20,
        n_heads: int = 10,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads  # 192
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(dim * 2.5))
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.embed.weight
        
        # ✅ BUG FIX #1: head_dim 기준 RoPE 계산
        f_cos, f_sin = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _get_freqs(self, f: torch.Tensor, start: int, length: int) -> torch.Tensor:
        return f[start:start + length]
    
    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s = tokens.shape
        x = self.embed(tokens)
        
        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]
        
        # ✅ BUG FIX #1, #2: head_dim 기준 RoPE 슬라이싱
        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.training:
                x, kv = checkpoint(
                    layer, x, f_cos, f_sin, None,
                    use_reentrant=False
                )
            else:
                kv_cache = kv_caches[i] if kv_caches else None
                x, kv = layer(x, f_cos, f_sin, kv_cache=kv_cache)
            
            new_kv_caches.append(kv)
        
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            # ✅ BUG FIX #4: Loss 계산 정확화
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=self.pad_token_id,
                reduction='mean'
            )
        
        return logits, loss, new_kv_caches

# ==========================================
# 3. 생성 함수
# ==========================================

@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str = "안녕? 너는 누구니?",
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,  # 1.0보다 크면 반복을 억제합니다.
    device: torch.device = None
) -> str:
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    prompt_text = f"### 지시: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    kv_caches = None
    output_tokens = tokens
    
    for step in range(max_tokens):
        input_tokens = output_tokens[:, -1:] if kv_caches is not None else output_tokens
        
        with torch.no_grad():
            logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)
        
        next_logits = logits[:, -1, :] / temperature
        
        # --- 반복 페널티 적용 시작 ---
        if repetition_penalty != 1.0:
            for token_id in set(output_tokens[0].tolist()):
                if next_logits[0, token_id] < 0:
                    next_logits[0, token_id] *= repetition_penalty
                else:
                    next_logits[0, token_id] /= repetition_penalty
        # --- 반복 페널티 적용 끝 ---
        
        # top_k filtering
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_logits, dim=-1)
        
        # top_p sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            probs[indices_to_remove] = 0.0
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        next_token = torch.multinomial(probs, num_samples=1)
        output_tokens = torch.cat([output_tokens, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        if output_tokens.shape[1] > 512:
            break
    
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = generated_text.split("### 응답:")[-1].strip() if "### 응답:" in generated_text else generated_text
    
    model.train()
    return response

# ==========================================
# 4. 체크포인트 저장/로드
# ==========================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_path: str
):
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device
) -> int:
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Model state loaded")
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✅ Optimizer state loaded")
        
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("✅ Scheduler state loaded")
        
        start_step = checkpoint['step']
        logger.info(f"✅ Checkpoint loaded from step {start_step}")
        
        return start_step
    
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return 0

def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    latest = checkpoints[-1]
    latest_path = os.path.join(checkpoint_dir, latest)
    logger.info(f"Found latest checkpoint: {latest}")
    
    return latest_path

# ==========================================
# GUI: Loss 그래프 + 채팅 창
# ==========================================

class TrainingMonitorGUI:
    """학습 모니터링 창 (Loss 그래프 + 현재 체크포인트 채팅)"""
    
    def __init__(self, tokenizer, device, model_config: dict):
        self.tokenizer = tokenizer
        # 채팅 모델은 항상 CPU에서 돌림
        self.device = torch.device("cpu") 
        self.model_config = model_config
        self.chat_model = None  # 채팅용 별도 모델 (최신 체크포인트 로드)
        self.current_ckpt_path = None
        self.running = True
        
        # 메시지 큐 (메인 스레드 → GUI)
        self.msg_queue = queue.Queue()
        
        self.root = tk.Tk()
        self.root.title("KoreanLLM Training Monitor 📊 + Chat")
        self.root.geometry("1200x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # 왼쪽: 그래프
        left_frame = Frame(self.root, width=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        Label(left_frame, text="📉 Loss Curve (실시간)", font=("Arial", 12, "bold")).pack()
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ax.grid(True, alpha=0.3)
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, label="Loss")
        self.ax.legend()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 오른쪽: 채팅
        right_frame = Frame(self.root, width=550)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ckpt_label = Label(right_frame, text="현재 체크포인트: (아직 없음)", font=("Arial", 10), fg="blue")
        self.ckpt_label.pack(pady=5)
        
        Label(right_frame, text="💬 모델과 대화하기 (CPU 구동)", font=("Arial", 12, "bold")).pack()
        
        self.chat_display = scrolledtext.ScrolledText(right_frame, height=25, width=60, state='disabled', wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        input_frame = Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.user_input = Entry(input_frame, font=("Arial", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", self.send_message)
        
        send_btn = Button(input_frame, text="전송", command=self.send_message, width=8)
        send_btn.pack(side=tk.RIGHT)
        
        refresh_btn = Button(right_frame, text="🔄 최신 체크포인트 로드", command=self.load_latest_checkpoint)
        refresh_btn.pack(pady=5)
        
        # 주기적 업데이트
        self.root.after(1000, self.update_gui)
    
    def on_close(self):
        self.running = False
        self.root.destroy()
    
    def update_gui(self):
        """큐에서 메시지 받아서 처리 + 그래프 업데이트"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg["type"] == "loss":
                    self._update_plot()
                elif msg["type"] == "ckpt":
                    self.current_ckpt_path = msg["path"]
                    self.ckpt_label.config(text=f"현재 체크포인트: {Path(msg['path']).name}")
                elif msg["type"] == "log":
                    self._append_chat(f"[시스템] {msg['text']}\n", "system")
        except queue.Empty:
            pass
        
        if self.running:
            self.root.after(1000, self.update_gui)
    
    def _update_plot(self):
        if not loss_history:
            return
        steps = [h["step"] for h in loss_history]
        losses = [h["loss"] for h in loss_history]
        self.line.set_data(steps, losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
    
    def _append_chat(self, text: str, tag: str = "user"):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, text)
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def load_latest_checkpoint(self):
        """최신 체크포인트를 채팅용 모델(CPU)에 로드"""
        ckpt = find_latest_checkpoint()
        if not ckpt:
            self._append_chat("[시스템] 체크포인트가 아직 없습니다.\n", "system")
            return
        
        try:
            self._append_chat(f"[시스템] 체크포인트 로딩 중(CPU): {Path(ckpt).name} ...\n", "system")
            self.root.update()
            
            # 채팅용 모델 새로 생성 (학습 모델과 분리, 무조건 CPU)
            if self.chat_model is None:
                self.chat_model = KoreanLLM(**self.model_config).to(self.device)
            
            checkpoint = torch.load(ckpt, map_location=self.device)
            self.chat_model.load_state_dict(checkpoint['model_state_dict'])
            self.chat_model.eval()
            
            self.current_ckpt_path = ckpt
            self.ckpt_label.config(text=f"현재 체크포인트: {Path(ckpt).name}")
            self._append_chat(f"[시스템] 로드 완료(CPU)! 이제 대화할 수 있어요.\n", "system")
        except Exception as e:
            self._append_chat(f"[시스템] 로드 실패: {e}\n", "system")
    
    def send_message(self, event=None):
        prompt = self.user_input.get().strip()
        if not prompt:
            return
        
        self.user_input.delete(0, tk.END)
        self._append_chat(f"나: {prompt}\n", "user")
        
        if self.chat_model is None:
            self._append_chat("[시스템] 먼저 '최신 체크포인트 로드' 버튼을 눌러주세요.\n", "system")
            return
        
        try:
            self._append_chat("모델(CPU): 생각 중...\n", "model")
            self.root.update()
            
            # CPU 장치를 명시적으로 전달하여 생성
            response = generate(
                self.chat_model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=80,
                temperature=0.7,
                top_p=0.95,
                device=self.device
            )
            
            # 마지막 "생각 중..." 줄 지우고 실제 응답 넣기
            self.chat_display.config(state='normal')
            self.chat_display.delete("end-2l", "end-1l")
            self.chat_display.config(state='disabled')
            
            self._append_chat(f"모델: {response}\n\n", "model")
        except Exception as e:
            self._append_chat(f"[시스템] 생성 오류: {e}\n", "system")
    
    def notify_loss(self):
        self.msg_queue.put({"type": "loss"})
    
    def notify_checkpoint(self, path: str):
        self.msg_queue.put({"type": "ckpt", "path": path})
    
    def notify_log(self, text: str):
        self.msg_queue.put({"type": "log", "text": text})
    
    def run(self):
        self.root.mainloop()

# 전역 GUI 인스턴스
gui_monitor: Optional[TrainingMonitorGUI] = None

# ==========================================
# 5. 메인 학습 루프
# ==========================================

@dataclass
class TrainingConfig:
    """학습 설정"""
    batch_size: int = 2
    max_steps: int = 50000
    accumulation_steps: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    checkpoint_interval: int = 100
    eval_interval: int = 500
    max_seq_len: int = 256
    num_workers: int = 4
    use_bfloat16: bool = True
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    download_datasets: bool = False
    samples_per_dataset: Optional[int] = None

def setup_distributed(rank: int = 0, world_size: int = 1):
    """분산학습 설정"""
    random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)

def main(config: TrainingConfig = TrainingConfig()):
    global gui_monitor
    
    setup_distributed()
    ensure_datasets_dir()
    load_loss_history()  # 이전 loss 기록 불러오기
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"📂 Datasets directory: {DATASETS_DIR.absolute()}")
    logger.info(f"📂 Logs directory: {LOG_DIR.absolute()}")
    
    # ============================================
    # 1. 토크나이저 로드
    # ============================================
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )
    
    # ✅ BUG FIX #7: EOS/PAD 토큰 명시적 설정
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Tokenizer: vocab_size={len(tokenizer)}, eos_id={tokenizer.eos_token_id}, pad_id={tokenizer.pad_token_id}")
    
    # ============================================
    # 2. 데이터셋 다운로드 및 로드
    # ============================================
    logger.info("Setting up datasets...")
    manager = DatasetManager()
    
    dataset_paths = manager.get_or_download_all(force=config.download_datasets)
    
    if not dataset_paths:
        logger.error("❌ No datasets available!")
        return
    
    dataset = LocalKoreanDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        data_samples_per_dataset=config.samples_per_dataset
    )
    
    if len(dataset) == 0:
        logger.error("❌ Dataset is empty!")
        return
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # ============================================
    # 3. 모델 생성
    # ============================================
    logger.info("Creating model...")
    model_config = dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1920,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    )
    model = KoreanLLM(**model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")
    
    # ============================================
    # GUI 시작 (별도 스레드)
    # ============================================
    def start_gui():
        global gui_monitor
        gui_monitor = TrainingMonitorGUI(tokenizer, device, model_config)
        gui_monitor.run()
    
    gui_thread = threading.Thread(target=start_gui, daemon=True)
    gui_thread.start()
    time.sleep(1.5)  # GUI가 뜰 시간 줌
    logger.info("🖥️  Monitoring GUI started (Loss graph + Chat)")
    
    # ============================================
    # 4. 옵티마이저와 스케줄러
    # ============================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # ✅ BUG FIX #9: Scheduler 타이밍 정확화
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # ============================================
    # 5. 체크포인트 로드
    # ============================================
    start_step = 0
    
    if config.resume_from_checkpoint:
        checkpoint_path = config.resume_from_checkpoint
        
        if checkpoint_path.lower() == 'latest':
            checkpoint_path = find_latest_checkpoint()
            if checkpoint_path is None:
                logger.warning("No checkpoint found, starting from scratch")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"🔄 Loading checkpoint from: {checkpoint_path}")
            start_step = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                device
            )
            if gui_monitor:
                gui_monitor.notify_checkpoint(checkpoint_path)
    
    # ============================================
    # 6. 학습 루프
    # ============================================
    logger.info(f"🚀 Starting training from step {start_step}...")
    logger.info(f"📊 Dataset size: {len(dataset)} samples")
    logger.info(f"📊 Total batches per epoch: {len(loader)}")
    
    model.train()
    optimizer.zero_grad()
    
    # ✅ BUG FIX #6: Loss 로깅 정확화
    running_loss = 0.0
    step = 0  # 원시 이터레이션 카운트
    
    try:
        epoch = 0
        while True:
            epoch += 1
            logger.info(f"\n📍 Epoch {epoch}")
            
            for batch_idx, batch in enumerate(loader):
                actual_step = (step // config.accumulation_steps) + start_step
                
                if actual_step >= config.max_steps:
                    logger.info(f"Reached max steps ({config.max_steps}), stopping training")
                    break
                
                batch = batch.to(device)
                
                # Forward pass with AMP
                if device.type == 'cuda' and config.use_bfloat16:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        _, loss, _ = model(batch, labels=batch)
                        loss_scaled = loss / config.accumulation_steps
                    
                    scaler.scale(loss_scaled).backward()
                else:
                    _, loss, _ = model(batch, labels=batch)
                    loss_scaled = loss / config.accumulation_steps
                    loss_scaled.backward()
                
                # ✅ BUG FIX #6: 원본 loss 누적 (이중 계산 방지)
                running_loss += loss.item()
                
                # Progress indicator
                if step % 4 == 0:
                    print(".", end="", flush=True)
                
                # Gradient accumulation
                if (step + 1) % config.accumulation_steps == 0:
                    if device.type == 'cuda' and config.use_bfloat16:
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    if device.type == 'cuda' and config.use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    # ✅ BUG FIX #9: scheduler.step() 호출 (accumulation 후)
                    scheduler.step()
                    
                    # 로깅
                    actual_step = (step + 1) // config.accumulation_steps + start_step
                    avg_loss = running_loss / config.accumulation_steps  # ✅ 정확한 평균
                    lr = scheduler.get_last_lr()[0]
                    
                    log_msg = f"[Step {actual_step:5d}] Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/step: {config.batch_size * config.max_seq_len}"
                    print(f"\n{log_msg}")
                    logger.info(log_msg)
                    
                    # Loss 히스토리 저장
                    loss_history.append({
                        "step": actual_step,
                        "loss": float(avg_loss),
                        "lr": float(lr),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    save_loss_history()
                    
                    # GUI에 알림
                    if gui_monitor:
                        gui_monitor.notify_loss()
                    
                    running_loss = 0.0

                    # --------------------------------------------------
                    # ⏸️ 250스텝마다 5초간 휴식 추가
                    # --------------------------------------------------
                    if actual_step > 0 and actual_step % 250 == 0:
                        logger.info(f"⏸️ {actual_step}스텝 도달: 5초간 휴식합니다...")
                        time.sleep(5)
                    # --------------------------------------------------
                    
                    # 평가 및 저장
                    if actual_step % config.eval_interval == 0:
                        logger.info("\n📝 Generating samples...")
                        prompts = [
                            "한국의 수도는",
                            "인공지능이란",
                            "안녕?"
                        ]
                        for prompt in prompts:
                            response = generate(
                                model, tokenizer, prompt=prompt,
                                max_tokens=50, temperature=0.7, top_p=0.95, device=device
                            )
                            logger.info(f"  Q: {prompt}\n  A: {response}")
                        
                        # 체크포인트 저장
                        checkpoint_path = f"checkpoints/korean_llm_{actual_step:05d}.pth"
                        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)
                        
                        if gui_monitor:
                            gui_monitor.notify_checkpoint(checkpoint_path)
                            gui_monitor.notify_log(f"체크포인트 저장됨: {Path(checkpoint_path).name}")
                
                step += 1
            
            if actual_step >= config.max_steps:
                break
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        actual_step = (step // config.accumulation_steps) + start_step
        checkpoint_path = f"checkpoints/korean_llm_interrupted_{actual_step:05d}.pth"
        save_checkpoint(model, optimizer, scheduler, actual_step, checkpoint_path)
        if gui_monitor:
            gui_monitor.notify_checkpoint(checkpoint_path)
    
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    
    logger.info("🎉 Training completed!")
    save_loss_history()
    
    # GUI가 계속 열려있도록 대기
    if gui_monitor and gui_monitor.running:
        logger.info("GUI가 열려 있습니다. 창을 닫으면 종료됩니다.")
        while gui_monitor.running:
            time.sleep(1)

if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=2,
        accumulation_steps=32,
        max_steps=50000,
        warmup_steps=200,
        learning_rate=5e-5,
        eval_interval=1000,
        resume_from_checkpoint='latest' if find_latest_checkpoint() else None,
        download_datasets=False,
        samples_per_dataset=None
    )
    main(config)