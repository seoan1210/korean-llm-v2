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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 아키텍처
# ==========================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=freqs.dtype)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attention(nn.Module):
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
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s, d = x.shape
        
        q = self.wq(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        if kv_cache is not None:
            pk, pv = kv_cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        
        # 학습 중에는 KV cache 생성 안 함
        if self.training:
            new_kv = None
        else:
            new_kv = (k.detach(), v.detach())
        
        has_cache = (kv_cache is not None and kv_cache[0] is not None)
        should_use_causal = (mask is None and not has_cache and s > 1)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=should_use_causal
        )
        
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out), new_kv

class TransformerBlock(nn.Module):
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
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        normed_x = self.attention_norm(x)
        h, new_kv = self.attention(normed_x, f_cos, f_sin, kv_cache=kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv

class KoreanLLM(nn.Module):
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
        
        f_cos, f_sin = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.register_buffer("f_cos", f_cos)
        self.register_buffer("f_sin", f_sin)
        
        self._init_weights()
        self.output.weight = self.embed.weight   # weight tying (초기화 후)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _get_freqs(self, f: torch.Tensor, start: int, length: int) -> torch.Tensor:
        return f[start:start + length].unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        tokens: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        b, s = tokens.shape
        x = self.embed(tokens)
        
        start_pos = 0
        if kv_caches is not None and len(kv_caches) > 0 and kv_caches[0] is not None:
            start_pos = kv_caches[0][0].shape[2]
        
        f_cos = self._get_freqs(self.f_cos, start_pos, s)
        f_sin = self._get_freqs(self.f_sin, start_pos, s)
        
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            if self.training:
                # ✅ layer_ref로 캡처해서 매번 함수 생성 방지
                layer_ref = layer
                def custom_forward(x_in, cos_in, sin_in):
                    out, _ = layer_ref(x_in, cos_in, sin_in, None)
                    return out
                
                x = checkpoint(custom_forward, x, f_cos, f_sin, use_reentrant=False)
                new_kv_caches.append(None)
            else:
                kv_cache = kv_caches[i] if kv_caches else None
                x, kv = layer(x, f_cos, f_sin, kv_cache=kv_cache)
                new_kv_caches.append(kv)
        
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                labels[..., 1:].reshape(-1),
                ignore_index=self.pad_token_id
            )
        
        return logits, loss, new_kv_caches

# ==========================================
# 2. 데이터셋
# ==========================================

class MultiKoreanDataset(IterableDataset):
    def __init__(self, tokenizer, max_len: int = 256, samples_per_epoch: int = 50000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples_per_epoch = samples_per_epoch
        
        self.datasets_config = [
            {"name": "maywell/korean_textbooks", "config": "tiny-textbooks", "split": "train"},
            {"name": "squarelike/OpenOrca-gugugo-ko", "config": None, "split": "train"},
            {"name": "beomi/KoAlpaca-v1.1a", "config": None, "split": "train"}
        ]
        
        logger.info(f"MultiKoreanDataset initialized with max_len={max_len}, samples_per_epoch={samples_per_epoch}")
    
    def _load_dataset_iterator(self, config: dict):
        try:
            if config["config"]:
                ds = load_dataset(config["name"], config["config"], split=config["split"], streaming=True)
            else:
                ds = load_dataset(config["name"], split=config["split"], streaming=True)
            return iter(ds)
        except Exception as e:
            logger.warning(f"Failed to load {config['name']}: {e}")
            return None
    
    def __iter__(self):
        sample_count = 0
        
        while True:
            dataset_iters = []
            for config in self.datasets_config:
                it = self._load_dataset_iterator(config)
                if it is not None:
                    dataset_iters.append(it)
            
            if not dataset_iters:
                logger.error("No valid datasets available!")
                break
            
            epoch_samples = 0
            while epoch_samples < self.samples_per_epoch:
                if not dataset_iters:
                    break
                    
                dataset_iter = random.choice(dataset_iters)
                
                try:
                    item = next(dataset_iter)
                    text = self._extract_text(item)
                    
                    if not text or len(text) < 5:
                        continue
                    
                    encoded = self._tokenize(text)
                    
                    if encoded is not None:
                        yield torch.tensor(encoded, dtype=torch.long)
                        epoch_samples += 1
                        sample_count += 1
                        
                        if sample_count % 1000 == 0:
                            logger.info(f"Processed {sample_count} samples")
                
                except StopIteration:
                    for idx, it in enumerate(dataset_iters):
                        if it is dataset_iter:
                            new_iter = self._load_dataset_iterator(self.datasets_config[idx])
                            if new_iter:
                                dataset_iters[idx] = new_iter
                            else:
                                dataset_iters.pop(idx)
                            break
                
                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue
    
    def _extract_text(self, item: dict) -> str:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        
        instruction = item.get("instruction")
        output = item.get("output")
        
        if instruction and output:
            input_text = item.get("input", "")
            if input_text:
                instruction = f"{instruction}\n{input_text}"
            return f"### 지시: {instruction}\n### 응답: {output}"
        
        question = item.get("question")
        if question:
            response = item.get("response")
            if response:
                return f"### 질문: {question}\n### 응답: {response}"
            
            answer = item.get("answer")
            if answer:
                return f"### 질문: {question}\n### 답변: {answer}"
        
        return ""
    
    def _tokenize(self, text: str) -> Optional[List[int]]:
        try:
            text_with_eos = text + self.tokenizer.eos_token
            encoded = self.tokenizer.encode(
                text_with_eos,
                truncation=True,
                max_length=self.max_len
            )
            if len(encoded) < self.max_len:
                encoded += [self.tokenizer.pad_token_id] * (self.max_len - len(encoded))
            return encoded
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return None

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
    device: torch.device = None
) -> str:
    if device is None:
        device = next(model.parameters()).device
    
    was_training = model.training
    model.eval()
    
    try:
        prompt_text = f"### 지시: {prompt}\n### 응답:"
        tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        
        if tokens.shape[1] > 256:
            tokens = tokens[:, -256:]
        
        kv_caches = None
        output_tokens = tokens.clone()
        
        for step in range(max_tokens):
            input_tokens = output_tokens if kv_caches is None else output_tokens[:, -1:]
            
            logits, _, kv_caches = model(input_tokens, kv_caches=kv_caches)
            
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            output_tokens = torch.cat([output_tokens, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            if output_tokens.shape[1] > 512:
                logger.warning("Generated sequence too long, truncating")
                break
        
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response = generated_text.split("### 응답:")[-1].strip() if "### 응답:" in generated_text else generated_text
        return response
    
    finally:
        if was_training:
            model.train()
        else:
            model.eval()

# ==========================================
# 4. 학습
# ==========================================

@dataclass
class TrainingConfig:
    batch_size: int = 2
    max_steps: int = 5000
    accumulation_steps: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 200
    checkpoint_interval: int = 100
    eval_interval: int = 10
    max_seq_len: int = 256
    num_workers: int = 0
    use_bfloat16: bool = True
    seed: int = 42
    resume_from: Optional[str] = None

def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config: TrainingConfig = TrainingConfig()):
    setup_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "beomi/Llama-3-Open-Ko-8B",
        clean_up_tokenization_spaces=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print("Vocab:", len(tokenizer))
    print("PAD:", tokenizer.pad_token_id)
    print("EOS:", tokenizer.eos_token_id)
    
    logger.info("Creating model...")
    model = KoreanLLM(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        dim=1280,
        n_layers=20,
        n_heads=10,
        max_seq_len=config.max_seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {total_params / 1e6:.1f}M total params, {trainable_params / 1e6:.1f}M trainable")
    
    logger.info("Creating dataset...")
    dataset = MultiKoreanDataset(tokenizer, max_len=config.max_seq_len, samples_per_epoch=50000)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim == 1:
            no_decay.append(p)
        else:
            decay.append(p)
    
    # ✅ fused=True 시도 (지원되면 자동 적용)
    try:
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            fused=True
        )
        logger.info("✅ Using fused AdamW")
    except TypeError:
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.learning_rate
        )
        logger.info("Using standard AdamW (fused not supported)")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    start_step = 0
    if config.resume_from and os.path.exists(config.resume_from):
        logger.info(f"Resuming from {config.resume_from}")
        ckpt = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_step = ckpt.get('step', 0)
        logger.info(f"Resumed from step {start_step}")
    
    logger.info("🚀 Starting training...")
    model.train()
    optimizer.zero_grad()
    
    running_loss = 0.0
    micro_step = 0
    optimizer_step = start_step
    
    try:
        for batch in loader:
            if optimizer_step >= config.max_steps:
                logger.info(f"Reached max optimizer steps ({config.max_steps})")
                break
            
            batch = batch.to(device, non_blocking=True)
            
            if device.type == 'cuda' and config.use_bfloat16:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _, loss, _ = model(batch, labels=batch)
                    loss = loss / config.accumulation_steps
                loss.backward()
            else:
                _, loss_val, _ = model(batch, labels=batch)
                loss = loss_val / config.accumulation_steps
                loss.backward()
            
            running_loss += loss.item() * config.accumulation_steps
            
            if micro_step % 4 == 0:
                print(".", end="", flush=True)
            
            if (micro_step + 1) % config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                optimizer_step += 1
                
                avg_loss = running_loss / config.accumulation_steps
                lr = scheduler.get_last_lr()[0]
                
                effective_batch = config.batch_size * config.accumulation_steps
                tokens_per_step = effective_batch * config.max_seq_len
                
                print(f"\n[Step {optimizer_step:5d}] Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/step: {tokens_per_step:,}")
                
                running_loss = 0.0
                
                if optimizer_step % config.eval_interval == 0:
                    logger.info(f"\n📊 Step {optimizer_step} | Total tokens: {tokens_per_step * optimizer_step:,}")
                    logger.info("\n📝 Generating samples...")
                    for prompt in ["한국의 수도는", "인공지능이란", "좋은 날씨에는"]:
                        response = generate(model, tokenizer, prompt=prompt, max_tokens=50, temperature=0.7, device=device)
                        logger.info(f"  Q: {prompt}\n  A: {response}")
                
                if optimizer_step % config.checkpoint_interval == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    path = f"checkpoints/korean_llm_{optimizer_step:05d}.pth"
                    torch.save({
                        'step': optimizer_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, path)
                    logger.info(f"✅ Checkpoint saved: {path}")
            
            micro_step += 1
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted")
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    
    logger.info("🎉 Training completed!")

if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=2,
        accumulation_steps=32,
        max_steps=5000,
        warmup_steps=200,
        learning_rate=5e-5,
        eval_interval=100,
        checkpoint_interval=100,
        seed=42,
    )
    main(config)