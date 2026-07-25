import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.checkpoint import checkpoint
import random
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging

# ==========================================
# 로깅 설정
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 아키텍처 (개선된 버전)
# ==========================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE(Rotary Position Embedding) 주파수 사전계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE 적용"""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    """SwiGLU 활성화 함수: GLU 기반의 현대적 FFN"""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
    """Multi-Head Attention with KV-Cache 지원"""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
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
        """
        Args:
            x: (batch, seq_len, dim)
            f_cos, f_sin: RoPE 주파수
            kv_cache: 이전 KV 캐시 (k, v)
            mask: 어텐션 마스크
        
        Returns:
            output: (batch, seq_len, dim)
            new_kv: (k, v) 새로운 캐시
        """
        b, s, d = x.shape
        
        # QKV 계산
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        
        # RoPE 적용
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        # KV-Cache 병합 (추론 시)
        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        
        new_kv = (k.detach(), v.detach())
        
        # Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None and s > 1)
        )
        
        # Output projection
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv

class TransformerBlock(nn.Module):
    """Transformer 블록 (Attention + FFN)"""
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
        # Self-Attention with Pre-Norm
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        
        # FFN with Pre-Norm
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x, new_kv

class KoreanLLM(nn.Module):
    """한국어 LLM 모델"""
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        dim: int = 1280,
        n_layers: int = 20,
        n_heads: int = 10,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.dim = dim
        self.n_heads = n_heads
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, int(dim * 2.5))
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        
        # Weight tying (임베딩과 출력층 가중치 공유)
        self.output.weight = self.embed.weight
        
        # RoPE 주파수 사전계산
        f_cos, f_sin = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """xavier 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _get_freqs(self, f: torch.Tensor, start: int, length: int) -> torch.Tensor:
        """시작 위치와 길이에 따른 주파수 추출"""
        return f[start:start + length].unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            tokens: (batch, seq_len)
            labels: (batch, seq_len) for training
            kv_caches: 이전 KV 캐시 리스트
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar or None
            new_kv_caches: 새로운 KV 캐시 리스트
        """
        b, s = tokens.shape
        
        # Embedding
        x = self.embed(tokens)
        
        # KV-Cache 시작 위치 계산
        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0][0] is not None:
            start_pos = kv_caches[0][0].shape[2]
        
        # RoPE 주파수 추출
        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)
        
        # Transformer 레이어 통과
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.training:
                # 훈련 시: gradient checkpointing 사용 (메모리 절약)
                x, kv = checkpoint(
                    layer, x, f_cos, f_sin, None,
                    use_reentrant=False
                )
            else:
                # 추론 시: KV-Cache 사용
                kv_cache = kv_caches[i] if kv_caches else None
                x, kv = layer(x, f_cos, f_sin, kv_cache=kv_cache)
            
            new_kv_caches.append(kv)
        
        # Output
        x = self.norm(x)
        logits = self.output(x)
        
        # Loss 계산
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=self.pad_token_id
            )
        
        return logits, loss, new_kv_caches

# ==========================================
# 2. 개선된 데이터셋
# ==========================================

class MultiKoreanDataset(IterableDataset):
    """다중 한국어 데이터셋 (robust한 버전)"""
    
    def __init__(self, tokenizer, max_len: int = 256, samples_per_epoch: int = 10000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        
        # 데이터셋 메타데이터 (한 번만 로드)
        self.datasets_config = [
            {"name": "maywell/korean_textbooks", "config": "tiny-textbooks", "split": "train"},
            {"name": "squarelike/OpenOrca-gugugo-ko", "config": None, "split": "train"},
            {"name": "beomi/KoAlpaca-v1.1a", "config": None, "split": "train"}
        ]
        
        logger.info(f"MultiKoreanDataset initialized with max_len={max_len}")
    
    def _load_dataset_iterator(self, config: dict):
        """데이터셋 이터레이터 생성"""
        try:
            if config["config"]:
                ds = load_dataset(
                    config["name"],
                    config["config"],
                    split=config["split"],
                    streaming=True
                )
            else:
                ds = load_dataset(
                    config["name"],
                    split=config["split"],
                    streaming=True
                )
            return iter(ds)
        except Exception as e:
            logger.warning(f"Failed to load {config['name']}: {e}")
            return None
    
    def __iter__(self):
        """무한 이터레이터 - 에포크 개념 없음"""
        sample_count = 0
        
        while True:
            # 매 에포크마다 새로운 이터레이터 생성
            dataset_iters = []
            for config in self.datasets_config:
                it = self._load_dataset_iterator(config)
                dataset_iters.append(it)
            
            # 유효한 이터레이터만 필터링
            dataset_iters = [it for it in dataset_iters if it is not None]
            
            if not dataset_iters:
                logger.error("No valid datasets available!")
                break
            
            # 이 에포크에서 sample_per_epoch개 샘플 생성
            epoch_samples = 0
            while epoch_samples < self.samples_per_epoch:
                # 랜덤하게 데이터셋 선택
                dataset_iter = random.choice(dataset_iters)
                
                try:
                    item = next(dataset_iter)
                    text = self._extract_text(item)
                    
                    if not text or len(text) < 5:
                        continue
                    
                    # 토크나이징
                    encoded = self._tokenize(text)
                    
                    if encoded is not None:
                        yield torch.tensor(encoded, dtype=torch.long)
                        epoch_samples += 1
                        sample_count += 1
                        
                        if sample_count % 1000 == 0:
                            logger.info(f"Processed {sample_count} samples")
                
                except StopIteration:
                    # 이 데이터셋이 끝났으면 새로 로드
                    idx = dataset_iters.index(dataset_iter)
                    new_iter = self._load_dataset_iterator(self.datasets_config[idx])
                    if new_iter:
                        dataset_iters[idx] = new_iter
                    else:
                        dataset_iters.remove(dataset_iter)
                
                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue
    
    def _extract_text(self, item: dict) -> str:
        """아이템에서 텍스트 추출"""
        if "text" in item and item["text"]:
            return item["text"]
        elif "instruction" in item and "output" in item:
            return f"### 지시: {item['instruction']}\n### 응답: {item['output']}"
        elif "question" in item and "answer" in item:
            return f"### 질문: {item['question']}\n### 답변: {item['answer']}"
        return ""
    
    def _tokenize(self, text: str) -> Optional[List[int]]:
        """텍스트 토크나이징"""
        try:
            # EOS 토큰 추가
            text_with_eos = text + self.tokenizer.eos_token
            
            # 토크나이징
            encoded = self.tokenizer.encode(
                text_with_eos,
                truncation=True,
                max_length=self.max_len
            )
            
            # 패딩 (고정 길이로 맞추기)
            if len(encoded) < self.max_len:
                encoded += [self.tokenizer.pad_token_id] * (self.max_len - len(encoded))
            
            return encoded
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return None

# ==========================================
# 3. 생성 함수 (개선된 버전)
# ==========================================

@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer,
    prompt: str = "안녕? 너는 누구니?",
    max_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 40,
    device: torch.device = None
) -> str:
    """
    개선된 생성 함수
    
    Args:
        model: LLM 모델
        tokenizer: 토크나이저
        prompt: 프롬프트
        max_tokens: 최대 생성 토큰 수
        temperature: 온도 (낮을수록 결정론적)
        top_k: Top-K 샘플링
        device: 디바이스
    
    Returns:
        생성된 텍스트
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 프롬프트 토크나이징
    prompt_text = f"### 지시: {prompt}\n### 응답:"
    tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    kv_caches = None
    output_tokens = tokens
    
    for step in range(max_tokens):
        # 마지막 토큰만 입력 (KV-Cache 활용)
        input_tokens = output_tokens[:, -1:] if kv_caches is not None else output_tokens
        
        # 모델 추론
        with torch.no_grad():
            logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)
        
        # 다음 토큰 샘플링
        next_logits = logits[:, -1, :] / temperature
        
        # Top-K 필터링
        if top_k > 0:
            indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
            next_logits[indices_to_remove] = float('-inf')
        
        # 소프트맥스
        probs = F.softmax(next_logits, dim=-1)
        
        # 샘플링
        next_token = torch.multinomial(probs, num_samples=1)
        
        output_tokens = torch.cat([output_tokens, next_token], dim=1)
        
        # EOS 토큰이면 중단
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # 최대 시퀀스 길이 체크 (메모리 누수 방지)
        if output_tokens.shape[1] > 512:
            logger.warning("Generated sequence too long, truncating")
            break
    
    # 디코딩
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    response = generated_text.split("### 응답:")[-1].strip() if "### 응답:" in generated_text else generated_text
    
    model.train()
    return response

# ==========================================
# 4. 메인 학습 루프
# ==========================================

@dataclass
class TrainingConfig:
    """학습 설정"""
    batch_size: int = 2  # RTX 5090에서 충분함
    max_steps: int = 50000
    accumulation_steps: int = 32  # 유효 배치사이즈 = 2 * 32 = 64
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    checkpoint_interval: int = 100
    eval_interval: int = 10
    max_seq_len: int = 256
    num_workers: int = 0
    use_bfloat16: bool = True
    seed: int = 42

def setup_distributed(rank: int = 0, world_size: int = 1):
    """분산학습 설정 (향후 확장용)"""
    random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + rank)

def main(config: TrainingConfig = TrainingConfig()):
    """메인 학습 함수"""
    setup_distributed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 토크나이저 로드
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 생성
    logger.info("Creating model...")
    model = KoreanLLM(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1280,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    # 모델 크기 로그
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")
    
    # 데이터로더
    logger.info("Creating dataset...")
    dataset = MultiKoreanDataset(
        tokenizer,
        max_len=config.max_seq_len,
        samples_per_epoch=config.accumulation_steps * config.eval_interval * 10
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # 옵티마이저와 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 학습 루프
    logger.info("🚀 Starting training...")
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    step = 0
    
    try:
        for batch_idx, batch in enumerate(loader):
            if step >= config.max_steps:
                logger.info(f"Reached max steps ({config.max_steps}), stopping training")
                break
            
            batch = batch.to(device)
            
            # Forward pass with AMP
            if device.type == 'cuda' and config.use_bfloat16:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss, _ = model(batch, labels=batch)
                    loss = loss / config.accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                _, loss_val, _ = model(batch, labels=batch)
                loss = loss_val / config.accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * config.accumulation_steps
            
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
                scheduler.step()
                
                # 로깅
                actual_step = (step + 1) // config.accumulation_steps
                avg_loss = running_loss / config.accumulation_steps
                lr = scheduler.get_last_lr()[0]
                
                print(f"\n[Step {actual_step:5d}] Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/step: {config.batch_size * config.max_seq_len}")
                
                running_loss = 0.0
                
                # 평가 및 저장
                if actual_step % config.eval_interval == 0:
                    logger.info("\n📝 Generating samples...")
                    prompts = [
                        "한국의 수도는",
                        "인공지능이란",
                        "좋은 날씨에는"
                    ]
                    for prompt in prompts:
                        response = generate(
                            model, tokenizer, prompt=prompt,
                            max_tokens=50, temperature=0.7, device=device
                        )
                        logger.info(f"  Q: {prompt}\n  A: {response}")
                    
                    # 체크포인트 저장
                    checkpoint_path = f"korean_llm_{actual_step:05d}.pth"
                    torch.save({
                        'step': actual_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpoint_path)
                    logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
            
            step += 1
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    
    logger.info("🎉 Training completed!")

if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=2,
        accumulation_steps=32,
        max_steps=50000,
        warmup_steps=200,
        learning_rate=5e-5,
        eval_interval=10,
        checkpoint_interval=100
    )
    main(config)
