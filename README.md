# 🇰🇷 KoreanLLM Training Framework

> **한국어 대규모 언어 모델(LLM) 학습 및 추론 프레임워크**  
> PAD/EOS 분리, 자동 데이터셋 관리, 실시간 모니터링 GUI 포함  
> 최대 23GB VRAM 최적화 (RTX 4090/A6000 기준)

---

## 📊 프로젝트 개요

이 프로젝트는 **한국어 특화** LLM을 효율적으로 학습할 수 있는 완전한 프레임워크입니다.

### ✨ 주요 특징

| 기능 | 설명 |
|------|------|
| 🤖 **Transformer 기반 아키텍처** | RMSNorm, RoPE, SwiGLU, Multi-Head Attention |
| 📚 **자동 데이터셋 관리** | HuggingFace 다운로드 + 로컬 캐싱 + Parquet 최적화 |
| ⚡ **메모리 최적화** | Gradient Checkpointing, Gradient Accumulation, AMP (bfloat16) |
| 🎯 **PAD/EOS 분리** | 명시적 특수 토큰 처리로 자연스러운 응답 종료 |
| 📉 **실시간 모니터링** | Loss 그래프 + GUI 채팅 인터페이스 |
| 💾 **체크포인트 관리** | 자동 저장/복구, Resume 기능 |
| 🔄 **OpenOrca 우회** | 대용량 데이터셋 다운로드 안정화 전략 |

---

## 🖥️ 시스템 요구사항

### ✅ **필수 사양 (VRAM 23GB 기준 - 수정 없이 즉시 사용 가능)**

| 항목 | 권장사양 | 비고 |
|------|--------|------|
| **GPU** | RTX 4090 (24GB) / A6000 (48GB) / RTX 6000 Ada | **원활한 학습 가능** |
| **GPU VRAM** | **최소 23GB** | 현재 설정: `batch_size=2`, `dim=1920`, `n_layers=20` |
| **메모리** | 64GB 이상 권장 | 데이터 전처리용 |
| **저장공간** | 1TB 이상 | 데이터셋(~200GB) + 체크포인트(~10GB 단계별) |
| **연결** | 안정적인 인터넷 | 데이터셋 다운로드용 |

### 🔧 **조정 가능한 사양 (저사양에서 실행하는 경우)**

다음 중 하나 이상을 조정하면 더 낮은 VRAM에서도 실행 가능합니다:

```python
# TrainingConfig 수정 예시
config = TrainingConfig(
    batch_size=1,              # 기본: 2 → 1로 감소
    dim=1024,                  # 기본: 1920 (모델 생성 시 수정 필요)
    n_layers=12,               # 기본: 20 (모델 생성 시 수정 필요)
    accumulation_steps=16,     # 기본: 32 → 감소
    max_seq_len=128,           # 기본: 256 → 감소
    num_workers=2,             # 기본: 4 → 감소
)
```

| VRAM | 추천 설정 |
|------|---------|
| **23GB+** | 기본값 (변경 불필요) |
| **16GB** | `batch_size=1`, `dim=1024`, `n_layers=12` |
| **12GB** | `batch_size=1`, `dim=768`, `n_layers=8`, `max_seq_len=128` |

---

## 🐍 소프트웨어 요구사항

### Python & PyTorch 버전

```
Python:        3.10.0 이상 (권장: 3.11.x)
PyTorch:       2.1.0 이상 (권장: 2.3.x ~ 2.4.x)
CUDA:          11.8 이상 (권장: 12.1+)
cuDNN:         8.7 이상
```

* 최신 GPU는 Pytorch Nightly 버전 추천

### 설치 명령어

```bash
# Python 3.11 환경 생성 (Conda 사용 권장)
conda create -n korean-llm-v2 python=3.11 -y
conda activate korean-llm-v2

# PyTorch 설치 (CUDA 12.1 기준)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 의존성 설치
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
transformers==4.36.0
datasets==2.16.0
pandas==2.1.3
matplotlib==3.8.2
numpy==1.24.3
tqdm==4.66.1
```

---

## 📦 설치 가이드

### 1️⃣ 저장소 클론

```bash
git clone https://github.com/seoan1210/korean-llm-v2.git
cd korean-llm-v2
```

### 2️⃣ 환경 설정

```bash
# Conda 환경 생성
conda create -n korean-llm python=3.11 -y
conda activate korean-llm

# PyTorch 설치 (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 모든 의존성 설치
pip install transformers==4.36.0 datasets==2.16.0 pandas==2.1.3 matplotlib==3.8.2 tqdm==4.66.1
```

### 3️⃣ 디렉토리 구조 확인

```
korean-llm-v2/
├── korean_llm_advanced_v2.py           # 메인 학습 스크립트
├── README.md
├── checkpoints/                  # 모델 체크포인트 (자동 생성)
├── datasets/                     # 캐시된 데이터셋 (자동 생성)
│   ├── cache/                    # 다운로드된 파일
│   └── datasets_manifest.json    # 데이터셋 메타데이터
└── logs/                         # 학습 로그 및 손실 히스토리 (자동 생성)
    ├── training.log              # 텍스트 로그
    └── loss_history.json         # JSON 손실 기록
```

---

## 🚀 빠른 시작

### 기본 실행 (권장)

```bash
# 모든 기본값 사용 (변경 불필요)
python korean_llm_advanced_v2.py
```

**일어나는 일:**
1. ✅ 토크나이저 로드 (beomi/Llama-3-Open-Ko-8B)
2. ✅ 데이터셋 자동 다운로드 (없으면)
3. ✅ 학습 시작 (체크포인트에서 자동 복구)
4. ✅ GUI 창 열림 (Loss 그래프 + 채팅)

---

## 📊 상세 사용 가이드

### 학습 설정 커스터마이징

`korean_llm_advanced_v2`의 **마지막 부분** 수정:

```python
if __name__ == "__main__":
    config = TrainingConfig(
        # ========== 메모리 설정 ==========
        batch_size=2,                    # GPU 배치 크기 (VRAM 충분하면 증가)
        accumulation_steps=32,           # 그래디언트 누적 스텝
        
        # ========== 학습 설정 ==========
        max_steps=50000,                 # 총 학습 스텝
        learning_rate=5e-5,              # 학습률
        warmup_steps=200,                # Warmup 스텝
        
        # ========== 데이터 설정 ==========
        max_seq_len=256,                 # 시퀀스 길이
        num_workers=4,                   # 데이터 로더 워커
        samples_per_dataset=None,        # 데이터셋당 샘플 수 (None=전체)
        
        # ========== 저장 및 평가 ==========
        checkpoint_interval=100,         # 체크포인트 저장 간격
        eval_interval=1000,              # 평가 및 샘플 생성 간격
        
        # ========== 기타 ==========
        use_bfloat16=True,               # 혼합 정밀도 (권장)
        resume_from_checkpoint='latest', # 마지막 체크포인트에서 복구
        download_datasets=False,         # 데이터셋 강제 재다운로드
    )
    main(config)
```

### 저사양 환경에서 실행

```python
# 예: RTX 3060 (12GB VRAM)
config = TrainingConfig(
    batch_size=1,                  # 배치 크기 감소
    accumulation_steps=16,         # 누적 스텝 감소
    max_seq_len=128,               # 시퀀스 길이 감소
    num_workers=2,                 # 워커 감소
    learning_rate=5e-5,
    warmup_steps=100,
    max_steps=30000,
)
```

### 새로운 학습 시작 (체크포인트 무시)

```python
config = TrainingConfig(
    # ... (다른 설정들)
    resume_from_checkpoint=None,   # 기존 체크포인트 로드 안 함
)
```

---

## 📚 데이터셋 상세 정보

### 사용되는 데이터셋

프로젝트에서는 **3개의 고품질 한국어 데이터셋**을 사용합니다:

#### 1️⃣ 한국어 교과서 데이터셋
- **이름**: maywell/korean_textbooks
- **크기**: ~50,000 문장
- **구성**: `text` 필드
- **특징**: 정식 한국어 교과서 내용
- **라이선스**: CC-BY-NC
- **링크**: https://huggingface.co/datasets/maywell/korean_textbooks

```python
# 데이터 예시
{
    "text": "한국은 동아시아에 위치한 나라입니다..."
}
```

#### 2️⃣ OpenOrca 한국어 번역 데이터셋
- **이름**: squarelike/OpenOrca-gugugo-ko
- **크기**: ~120,000 Q&A 쌍
- **구성**: `question`, `response` 필드
- **특징**: 다양한 주제의 대화형 데이터, 한국어 번역
- **라이선스**: OpenRAIL (상업 사용 가능)
- **링크**: https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko
- **원본 논문**: [OpenOrca: An Open Research Corpus for ChatGPT (Liang et al., 2023)](https://arxiv.org/abs/2310.02959)

```python
# 데이터 예시
{
    "system_prompt": "당신은 도움이 되는 AI 어시스턴트입니다.",
    "question": "한국의 수도는?",
    "response": "한국의 수도는 서울입니다."
}
```

#### 3️⃣ KoAlpaca 한국어 명령어 데이터셋
- **이름**: beomi/KoAlpaca-v1.1a
- **크기**: ~52,000 지시-응답 쌍
- **구성**: `instruction`, `output`, `input` 필드
- **특징**: Alpaca 포맷의 지시 튜닝 데이터
- **라이선스**: CC-BY-NC-4.0
- **링크**: https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a
- **참고논문**: [Stanford Alpaca (Taori et al., 2023)](https://github.com/tatsu-lab/stanford_alpaca)

```python
# 데이터 예시
{
    "instruction": "다음 문장을 요약해주세요.",
    "input": "한국 경제는 최근 회복세를 보이고 있습니다...",
    "output": "한국 경제가 회복 중입니다."
}
```

### 데이터셋 다운로드 및 캐싱

**첫 실행 시 자동 다운로드:**
```bash
python korean_llm_fixed.py
# → datasets/ 디렉토리에 자동으로 다운로드되고 parquet 형식으로 캐싱됨
```

**다운로드 진행 상황:**
```
📥 Downloading maywell/korean_textbooks...
🔄 Attempt 1/3 - Strategy: standard
✅ Dataset saved: ./datasets/cache/a1b2c3d4 (50000 examples)

📥 Downloading squarelike/OpenOrca-gugugo-ko...
🔄 Attempt 1/3 - Strategy: parquet_auto
...
```

**캐시 위치:**
```
datasets/
├── cache/
│   ├── a1b2c3d4/               # korean_textbooks
│   │   └── data.parquet
│   ├── b2c3d4e5/               # OpenOrca-gugugo-ko
│   │   └── data.parquet
│   └── c3d4e5f6/               # KoAlpaca-v1.1a
│       └── data.parquet
└── datasets_manifest.json      # 메타데이터
```

### 데이터 포맷 통일

모든 데이터셋은 다음 포맷으로 자동 변환됩니다:

```
### 질문: [질문 텍스트]
### 응답: [응답 텍스트]
```

**변환 규칙:**
```python
# 1. text 필드 → 그대로 사용
if "text" in item:
    text = item["text"]

# 2. instruction + output → 질문-응답 형식으로 변환
elif "instruction" in item and "output" in item:
    text = f"### 질문: {item['instruction']}\n### 응답: {item['output']}"

# 3. question + response → 질문-응답 형식으로 변환
elif "question" in item and "response" in item:
    text = f"### 질문: {item['question']}\n### 응답: {item['response']}"

# 4. question + answer → 질문-응답 형식으로 변환
elif "question" in item and "answer" in item:
    text = f"### 질문: {item['question']}\n### 응답: {item['answer']}"

# 5. prompt + response → 질문-응답 형식으로 변환
elif "prompt" in item and "response" in item:
    text = f"### 질문: {item['prompt']}\n### 응답: {item['response']}"
```

---

## 🧠 모델 아키텍처

### 기본 설정 (23GB VRAM & 64GB RAM)

```
┌─────────────────────────────────────────────┐
│ KoreanLLM Transformer                       │
├─────────────────────────────────────────────┤
│ Vocabulary Size:     128,256 (Llama-3)      │
│ Hidden Dimension:    1,920                  │
│ Number of Layers:    20                     │
│ Number of Heads:     10                     │
│ Head Dimension:      192                    │
│ Total Parameters:    ~3.6B                  │
│ Max Sequence Length: 2,048 (학습: 256)      │
└─────────────────────────────────────────────┘
```

### 아키텍처 구성요소

| 컴포넌트 | 설명 | 논문 |
|---------|------|------|
| **RMSNorm** | Root Mean Square Layer Normalization | [T5 (Raffel et al., 2020)](https://arxiv.org/abs/1910.10683) |
| **RoPE** | Rotary Position Embedding | [RoPE (Su et al., 2021)](https://arxiv.org/abs/2104.09864) |
| **SwiGLU** | Swish Gated Linear Unit | [GLU Variants (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) |
| **Multi-Head Attention** | Scaled Dot-Product Attention | [Transformer (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) |
| **KV-Cache** | Key-Value Cache for Inference | [Llama (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) |
| **Gradient Checkpointing** | 메모리 효율 최적화 | [PyTorch 공식문서](https://pytorch.org/docs/stable/checkpoint.html) |

### 학습 기법

| 기법 | 설정값 | 이유 |
|------|--------|------|
| **Batch Size** | 2 | VRAM 최적화 |
| **Gradient Accumulation** | 32 steps | Effective batch = 2 × 32 = 64 |
| **Mixed Precision** | bfloat16 | VRAM 절약 + 안정성 |
| **Gradient Clipping** | 1.0 | 학습 안정성 |
| **Optimizer** | AdamW | 범용 최적화 알고리즘 |
| **Learning Rate** | 5e-5 | Warmup 후 Cosine 감소 |
| **Warmup Steps** | 200 | 초기 불안정성 완화 |
| **Max Steps** | 50,000 | ~50만 배치 학습 |

---

## 📈 학습 프로세스

### 손실 함수 (Loss Function)

```python
# Cross-Entropy Loss (예측과 실제의 차이 계산)
loss = F.cross_entropy(
    logits[..., :-1, :].reshape(-1, vocab_size),
    labels[..., 1:].reshape(-1),
    ignore_index=pad_token_id,  # PAD 토큰 제외
    reduction='mean'
)
```

**특징:**
- ✅ EOS 토큰 학습 (자연스러운 응답 종료)
- ✅ PAD 토큰 무시 (의미 없는 패딩 학습 방지)
- ✅ Next-token prediction 방식

### 학습 스케줄

```
Step: 0 ──→ 200 (Warmup) ──→ 50,000 (Cosine Decay) ──→ End
LR:   0 ──→ 5e-5 ────────→ 5e-5 ──→ ~1e-6 ────────→ ~0
```

**Cosine Annealing Warmup:**
- 초기 200 스텝: Learning Rate 선형 증가 (0 → 5e-5)
- 200~50,000 스텝: 코사인 함수로 감소

### 체크포인트 저장

**저장 주기:**
- **Eval Interval**: 1,000 스텝마다
- **저장 위치**: `checkpoints/korean_llm_{step:05d}.pth`

**저장 내용:**
```python
{
    'step': int,                          # 학습 스텝
    'model_state_dict': dict,             # 모델 가중치
    'optimizer_state_dict': dict,         # 옵티마이저 상태
    'scheduler_state_dict': dict,         # 스케줄러 상태
}
```

**예시:**
```
checkpoints/
├── korean_llm_01000.pth
├── korean_llm_02000.pth
├── korean_llm_03000.pth
└── korean_llm_latest.pth  # 자동 링크
```

---

## 🎨 GUI 모니터링

학습 중 자동으로 GUI 창이 열립니다.

### 좌측: Loss 그래프

```
📉 Loss Curve (실시간)
┌──────────────────────────────┐
│                        ▀▄     │ Loss
│                         ▄▀   │
│      ▀▀▀▄▄▄▄▄▄▄▄▀▀▀▀▀     │
│                             │
└──────────────────────────────┘
0              Steps            50000
```

**특징:**
- 실시간 업데이트
- 학습 진행상황 시각화
- 이상 감지 용이

### 우측: 채팅 인터페이스

```
현재 체크포인트: korean_llm_05000.pth
💬 모델과 대화하기 (CPU 구동)

┌──────────────────────────┐
│ 나: 한국의 수도는?       │
│ 모델: 서울입니다.        │
│                          │
│ 나: 인공지능이란?        │
│ 모델: 인공지능은...      │
└──────────────────────────┘

[입력창] [전송 버튼] [🔄 최신 체크포인트 로드]
```

**기능:**
- ✅ 최신 체크포인트 자동 로드 (CPU)
- ✅ 학습 중 모델 테스트 가능
- ✅ 대화형 인터페이스
- ✅ 모델 이해도 평가

### 샘플 생성

1,000 스텝마다 자동으로 다음 프롬프트에 대한 답변 생성:

```
Q: 한국의 수도는
A: [모델이 생성한 답변]

Q: 인공지능이란
A: [모델이 생성한 답변]

Q: 안녕?
A: [모델이 생성한 답변]
```

---

## 📝 로깅 및 모니터링

### 로그 파일 위치

```
logs/
├── training.log              # 텍스트 로그 (실시간)
└── loss_history.json         # JSON 손실 기록 (분석용)
```

### training.log 예시

```
2024-01-15 10:30:45,123 - INFO - Using device: cuda
2024-01-15 10:30:46,234 - INFO - Loading tokenizer...
2024-01-15 10:30:47,345 - INFO - ✅ Added separate pad token: <|pad|>
2024-01-15 10:30:48,456 - INFO - Tokenizer: vocab_size=128257, eos_id=128001, pad_id=128256
2024-01-15 10:30:49,567 - INFO - 📚 Loading local datasets...
2024-01-15 10:30:50,678 - INFO - ✅ Loaded 50000 samples from cache/a1b2c3d4
2024-01-15 10:30:51,789 - INFO - ✅ Total samples loaded: 222000

2024-01-15 10:31:00,000 - INFO - Creating model...
2024-01-15 10:31:02,000 - INFO - Model: 3600.0M total params, 3600.0M trainable

[Step  1000] Loss: 3.4521 | LR: 4.95e-05 | Tokens/step: 512
[Step  2000] Loss: 3.2145 | LR: 4.90e-05 | Tokens/step: 512
[Step  3000] Loss: 2.9876 | LR: 4.85e-05 | Tokens/step: 512

📝 Generating samples...
  Q: 한국의 수도는
  A: 한국의 수도는 서울입니다. 서울은 한반도의 중서부에 위치하고...

✅ Checkpoint saved: checkpoints/korean_llm_01000.pth
```

### loss_history.json 예시

```json
[
  {
    "step": 100,
    "loss": 3.6234,
    "lr": 2.5e-5,
    "time": "2024-01-15 10:31:15"
  },
  {
    "step": 200,
    "loss": 3.5120,
    "lr": 5.0e-5,
    "time": "2024-01-15 10:31:30"
  },
  ...
]
```

**분석 용도:**
```python
import json
import matplotlib.pyplot as plt

with open('logs/loss_history.json', 'r') as f:
    history = json.load(f)

steps = [h['step'] for h in history]
losses = [h['loss'] for h in history]

plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('loss_curve.png')
```

---

## 🔧 고급 사용법

### 특정 스텝부터 학습 재개

```bash
# 5000번째 스텝부터 재개
python korean_llm_fixed.py  # resume_from_checkpoint='latest' 설정 유지
```

### 학습 중단 및 복구

```bash
# 학습 중 Ctrl+C 누르면:
# → interrupted_XXXXX.pth 체크포인트 자동 저장
# → 다시 실행하면 그 지점에서 복구

python korean_llm_fixed.py  # 자동으로 latest 로드
```

### 데이터셋 재다운로드 강제

```python
config = TrainingConfig(
    download_datasets=True,  # 기존 캐시 무시하고 재다운로드
    # ... 다른 설정
)
```

### 데이터셋 부분 학습

```python
config = TrainingConfig(
    samples_per_dataset=5000,  # 각 데이터셋에서 5,000개씩만 사용
    # ... 다른 설정
)
```

### 추론 전용 모드 (생성만 하기)

```python
from korean_llm_fixed import KoreanLLM, generate, AutoTokenizer
import torch

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
model = KoreanLLM().to(device)

# 체크포인트 로드
checkpoint = torch.load("checkpoints/korean_llm_05000.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 생성
with torch.no_grad():
    response = generate(
        model, tokenizer,
        prompt="한국의 문화는",
        max_tokens=100,
        temperature=0.7,
        top_p=0.95,
        device=device
    )
    print(f"Q: 한국의 문화는\nA: {response}")
```

---

## 📊 성능 예상

### 학습 시간 (NVIDIA RTX 4090, 23GB VRAM)

| 설정 | Tokens/sec | 시간/1K 스텝 | 50K 스텝 예상시간 |
|------|-----------|-----------|-----------------|
| batch_size=2, bfloat16 | ~2,000 | ~5분 | ~4시간 |
| batch_size=4, bfloat16 | ~3,500 | ~3분 | ~2.5시간 |

### 메모리 사용량

| 항목 | 메모리 |
|------|--------|
| 모델 가중치 | ~7.2GB |
| 옵티마이저 상태 | ~14.4GB |
| 활성화 (bfloat16) | ~1.0GB |
| 기타 오버헤드 | ~0.4GB |
| **합계** | **~23GB** |

### 생성 속도 (추론)

| 설정 | Tokens/sec | 100 Token 생성 시간 |
|------|-----------|------------------|
| Temperature=0.7 | ~100 | ~1초 |
| Temperature=0.5 | ~100 | ~1초 |
| Top-p=0.95 | ~90 | ~1.1초 |

---

## 🐛 트러블슈팅

### 문제 1: CUDA Out of Memory

**증상:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB.
```

**해결책:**
```python
# 옵션 1: 배치 크기 감소
config = TrainingConfig(batch_size=1, ...)

# 옵션 2: 시퀀스 길이 감소
config = TrainingConfig(max_seq_len=128, ...)

# 옵션 3: 누적 스텝 감소
config = TrainingConfig(accumulation_steps=16, ...)

# 옵션 4: 모델 크기 감소 (korean_llm_fixed.py 수정 필요)
# → dim=1024, n_layers=12 로 변경
```

### 문제 2: 데이터셋 다운로드 실패

**증상:**
```
❌ Failed to download squarelike/OpenOrca-gugugo-ko after 3 strategies
```

**해결책:**
```bash
# 1. 인터넷 연결 확인
ping huggingface.co

# 2. 캐시 삭제 후 재시도
rm -rf datasets/cache/*
python korean_llm_fixed.py

# 3. HuggingFace CLI 로그인
huggingface-cli login
```

### 문제 3: GPU 메모리 누수

**증상:**
```
GPU 메모리가 계속 증가하다가 OOM 에러
```

**해결책:**
```python
# 1. Gradient Checkpointing 활성화 확인 (기본값 활성화)
# 2. 워커 수 감소
config = TrainingConfig(num_workers=0, ...)

# 3. 주기적으로 GPU 캐시 초기화
import torch
torch.cuda.empty_cache()
```

### 문제 4: Loss가 감소하지 않음

**증상:**
```
[Step  1000] Loss: 3.4521
[Step  2000] Loss: 3.4120
[Step  3000] Loss: 3.4312  ← 감소 안 함
```

**해결책:**
```python
# 1. 학습률 증가
config = TrainingConfig(learning_rate=1e-4, ...)

# 2. Warmup 스텝 증가
config = TrainingConfig(warmup_steps=500, ...)

# 3. 데이터셋 확인
# → 동일한 데이터가 반복되거나 노이즈가 많으면 문제 발생 가능
```

### 문제 5: 모델이 학습되지 않은 것처럼 응답

**증상:**
```
Q: 한국의 수도는?
A: 한국 한국 한국 한국... (반복만 함)
```

**해결책:**
```python
# 1. 더 많은 스텝 학습 필요
config = TrainingConfig(max_steps=100000, ...)

# 2. EOS 페널티 추가 (생성 함수 수정)
response = generate(
    model, tokenizer,
    prompt="한국의 수도는",
    repetition_penalty=1.5,  # 반복 억제
    temperature=0.7,
)

# 3. 데이터셋 다양성 확인
# → 동일한 질문이 많으면 과적합 가능
```

---

## 📚 참고 자료 및 논문

### 핵심 논문

| 제목 | 저자 | 연도 | 링크 |
|------|------|------|------|
| **Attention is All You Need** | Vaswani et al. | 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| **RoPE: Rotary Position Embedding** | Su et al. | 2021 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| **Exploring the Limits of Transfer Learning** (T5/RMSNorm) | Raffel et al. | 2020 | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) |
| **GLU Variants Improve Transformer** (SwiGLU) | Shazeer | 2020 | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) |
| **Llama: Open and Efficient Foundation Language Models** | Touvron et al. | 2023 | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) |
| **Llama 2: Open Foundation and Fine-Tuned Chat Models** | Touvron et al. | 2023 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| **OpenOrca: An Open Research Corpus for ChatGPT** | Liang et al. | 2023 | [arXiv:2310.02959](https://arxiv.org/abs/2310.02959) |

### PyTorch & Transformers 문서

- **PyTorch 공식 문서**: https://pytorch.org/docs/stable/index.html
- **Gradient Checkpointing**: https://pytorch.org/docs/stable/checkpoint.html
- **Transformers 라이브러리**: https://huggingface.co/docs/transformers/
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/

### 한국어 NLP 자료

- **KoGPT2**: https://github.com/SKT-AI/KoGPT2
- **KoELECTRA**: https://github.com/monologg/KoELECTRA
- **Llama-3-Open-Ko**: https://huggingface.co/beomi/Llama-3-Open-Ko-8B

---

## 🤝 커뮤니티 및 지원

### 버그 리포트

GitHub Issues에 다음 정보와 함께 보고해주세요:

```
- Python 버전
- PyTorch 버전
- CUDA 버전
- GPU 모델 및 VRAM
- 정확한 에러 메시지
- 재현 가능한 코드
```

### 기여 방법

```bash
# 1. Fork
git clone https://github.com/yourname/korean-llm-training.git

# 2. Branch 생성
git checkout -b feature/your-feature

# 3. Commit
git commit -m "Add your feature"

# 4. Push
git push origin feature/your-feature

# 5. Pull Request 생성
```

---

## 📄 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다.

**사용된 데이터셋 라이선스:**
- maywell/korean_textbooks: CC-BY-NC
- squarelike/OpenOrca-gugugo-ko: OpenRAIL
- beomi/KoAlpaca-v1.1a: CC-BY-NC-4.0

각 데이터셋의 라이선스 조건을 확인하고 사용하세요.

---

## 📞 연락처

- **이메일**: your.email@example.com
- **GitHub**: https://github.com/yourname
- **HuggingFace**: https://huggingface.co/yourname

---

## 🎯 로드맵

### v1.0 (현재)
- ✅ 기본 학습 프레임워크
- ✅ GUI 모니터링
- ✅ 데이터셋 통합
- ✅ PAD/EOS 분리

### v1.1 (예정)
- ⏳ LoRA 파인튜닝 지원
- ⏳ Quantization (8-bit, 4-bit)
- ⏳ 분산 학습 (DDP)
- ⏳ Flash Attention V2

### v2.0 (미래)
- 📋 Mixture of Experts (MoE)
- 📋 Multi-GPU 분산학습
- 📋 Wandb 통합
- 📋 자동 하이퍼파라미터 튜닝

---

## 💡 팁 & 트릭

### 빠른 프로토타이핑

```python
# 10,000 스텝만 학습해보기
config = TrainingConfig(
    max_steps=10000,
    eval_interval=500,
    samples_per_dataset=10000,  # 전체 데이터의 일부만 사용
)
```

### 최대 성능 추출

```python
# RTX 4090에서 최대 성능
config = TrainingConfig(
    batch_size=4,              # 4로 증가
    accumulation_steps=16,     # 16으로 감소 (Effective: 64)
    num_workers=8,             # 8로 증가
    use_bfloat16=True,         # 유지
)
```

### 저전력 모드

```python
# 에너지 효율 최우선
config = TrainingConfig(
    batch_size=1,
    accumulation_steps=4,
    max_seq_len=128,
    num_workers=0,
)
```

---

## 🎓 학습 자료

### 초보자를 위한 가이드

1. **기본 개념 이해**
   - [What is a Language Model?](https://huggingface.co/course/chapter1/1)
   - [Transformer 아키텍처](https://jalammar.github.io/illustrated-transformer/)

2. **코드 실행**
   - 기본 설정으로 먼저 실행
   - logs/training.log 모니터링
   - GUI에서 손실 감소 확인

3. **커스터마이징**
   - 데이터셋 변경 시도
   - 하이퍼파라미터 조정
   - 결과 비교

### 고급 주제

1. **메모리 최적화**
   - Gradient Checkpointing의 원리
   - Mixed Precision Training
   - Activation Checkpointing

2. **성능 향상**
   - Knowledge Distillation
   - Quantization Aware Training
   - Pruning

---

## 🙏 감사의 말

이 프로젝트는 다음의 오픈소스 커뮤니티를 기반으로 합니다:

- **PyTorch**: 딥러닝 프레임워크
- **HuggingFace**: Transformers와 Datasets
- **Llama**: 메타 오픈소스 LLM
- **한국 NLP 커뮤니티**: 데이터셋 제공

---

**마지막 업데이트**: 2026년 8월 8일  
**유지보수자**: seoan1210
**상태**: 🟢 활발히 유지 중
