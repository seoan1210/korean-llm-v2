# 🇰🇷 KoreanLLM Training Framework

[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%2B-76b900?style=flat-square)](https://developer.nvidia.com/cuda-toolkit)
[![Status](https://img.shields.io/badge/Status-Active%20Training-brightgreen?style=flat-square)]()
[![Steps](https://img.shields.io/badge/Steps-2000%2F50000-orange?style=flat-square)]()
[![Loss](https://img.shields.io/badge/Loss-2.6491%20±%200.08-yellow?style=flat-square)]()

> **한국어 대규모 언어 모델(LLM) 학습 및 추론 프레임워크**  
> PAD/EOS 분리 · 자동 데이터셋 관리 · 실시간 모니터링 GUI  
> **v2** | 최대 23GB VRAM 최적화 (RTX 4090 / A6000 기준)

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

### ✅ 필수 사양 (VRAM 23GB 기준 – 수정 없이 즉시 사용 가능)

| 항목 | 권장사양 | 비고 |
|------|----------|------|
| **GPU** | RTX 4090 (24GB) / A6000 (48GB) / RTX 6000 Ada | 원활한 학습 가능 |
| **GPU VRAM** | **최소 23GB** | 현재 설정: `batch_size=2`, `dim=1920`, `n_layers=20` |
| **메모리** | 64GB 이상 권장 | 데이터 전처리용 |
| **저장공간** | 1TB 이상 | 데이터셋(~200GB) + 체크포인트(~10GB 단계별) |
| **연결** | 안정적인 인터넷 | 데이터셋 다운로드용 |

### 🔧 저사양 조정 예시

```python
config = TrainingConfig(
    batch_size=1,          # 기본: 2 → 1
    dim=1024,              # 기본: 1920 (모델 생성 시 수정)
    n_layers=12,           # 기본: 20 (모델 생성 시 수정)
    accumulation_steps=16, # 기본: 32
    max_seq_len=128,       # 기본: 256
    num_workers=2,         # 기본: 4
)
```

| VRAM | 추천 설정 |
|------|-----------|
| **23GB+** | 기본값 (변경 불필요) |
| **16GB** | `batch_size=1`, `dim=1024`, `n_layers=12` |
| **12GB** | `batch_size=1`, `dim=768`, `n_layers=8`, `max_seq_len=128` |

---

## 🐍 소프트웨어 요구사항

```
Python  : 3.10.0 이상 (권장 3.11.x)
PyTorch : 2.1.0 이상 (권장 2.3.x ~ 2.4.x)
CUDA    : 11.8 이상 (권장 12.1+)
cuDNN   : 8.7 이상
```

### 설치 명령어

```bash
# Conda 환경 생성
conda create -n korean-llm-v2 python=3.11 -y
conda activate korean-llm-v2

# PyTorch (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 의존성
pip install transformers==4.36.0 datasets==2.16.0 pandas==2.1.3 matplotlib==3.8.2 numpy==1.24.3 tqdm==4.66.1
```

---

## 📦 설치 가이드

### 1️⃣ 저장소 클론

```bash
git clone https://github.com/seoan1210/korean-llm-v2.git
cd korean-llm-v2
```

### 2️⃣ 환경 설정

위 설치 명령어 그대로 실행하면 됩니다.

### 3️⃣ 디렉토리 구조

```
korean-llm-v2/
├── .gitattributes
├── LICENSE
├── README.md
└── korean_llm_advanced_v2.py   # 메인 학습 스크립트
```

실행 시 자동 생성되는 폴더:

```
├── checkpoints/                # 모델 체크포인트
├── datasets/
│   ├── cache/                  # 다운로드된 parquet
│   └── datasets_manifest.json
└── logs/
    ├── training.log
    └── loss_history.json
```

---

## 🚀 빠른 시작

```bash
python korean_llm_advanced_v2.py
```

**실행되면 일어나는 일:**
1. 토크나이저 로드 (`beomi/Llama-3-Open-Ko-8B`)
2. 데이터셋 자동 다운로드 (없으면)
3. 학습 시작 (체크포인트 있으면 자동 복구)
4. GUI 창 열림 (Loss 그래프 + 채팅)

---

## 📊 학습 설정 커스터마이징

`korean_llm_advanced_v2.py` 맨 아래 부분을 수정하세요.

```python
if __name__ == "__main__":
    config = TrainingConfig(
        batch_size=2,
        accumulation_steps=32,
        max_steps=50000,
        learning_rate=5e-5,
        warmup_steps=200,
        max_seq_len=256,
        num_workers=4,
        samples_per_dataset=None,
        checkpoint_interval=100,
        eval_interval=1000,
        use_bfloat16=True,
        resume_from_checkpoint='latest',
        download_datasets=False,
    )
    main(config)
```

### 새로운 학습 시작 (체크포인트 무시)

```python
config = TrainingConfig(
    resume_from_checkpoint=None,
    # ... 나머지 설정
)
```

---

## 📚 데이터셋

프로젝트에서 사용하는 3개 데이터셋:

| 데이터셋 | 크기 | 필드 | 라이선스 |
|----------|------|------|----------|
| [maywell/korean_textbooks](https://huggingface.co/datasets/maywell/korean_textbooks) | ~50,000 | `text` | CC-BY-NC |
| [squarelike/OpenOrca-gugugo-ko](https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko) | ~120,000 | `question`, `response` | OpenRAIL |
| [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a) | ~52,000 | `instruction`, `output` | CC-BY-NC-4.0 |

모든 데이터는 아래 포맷으로 자동 변환됩니다:

```
### 질문: [질문 텍스트]
### 응답: [응답 텍스트]
```

첫 실행 시 `datasets/` 폴더에 parquet로 캐싱됩니다.

---

## 🧠 모델 아키텍처

```
┌─────────────────────────────────────────────┐
│              KoreanLLM Transformer          │
├─────────────────────────────────────────────┤
│ Vocabulary Size     : 128,256 (Llama-3)     │
│ Hidden Dimension    : 1,920                 │
│ Number of Layers    : 20                    │
│ Number of Heads     : 10                    │
│ Head Dimension      : 192                   │
│ Total Parameters    : ~3.6B                 │
│ Max Sequence Length : 2,048 (학습 시 256)    │
└─────────────────────────────────────────────┘
```

### 아키텍처 구성요소

| 컴포넌트 | 설명 | 참고 |
|----------|------|------|
| **RMSNorm** | Root Mean Square Layer Normalization | T5 |
| **RoPE** | Rotary Position Embedding | Su et al., 2021 |
| **SwiGLU** | Swish Gated Linear Unit | Shazeer, 2020 |
| **Multi-Head Attention** | Scaled Dot-Product Attention | Vaswani et al., 2017 |
| **KV-Cache** | Key-Value Cache (추론용) | Llama |
| **Gradient Checkpointing** | 메모리 효율 최적화 | PyTorch |

### 학습 기법

| 기법 | 설정값 | 비고 |
|------|--------|------|
| Batch Size | 2 | VRAM 최적화 |
| Gradient Accumulation | 32 steps | Effective batch = 64 |
| Mixed Precision | bfloat16 | VRAM 절약 + 안정성 |
| Gradient Clipping | 1.0 | 학습 안정성 |
| Optimizer | AdamW | - |
| Learning Rate | 5e-5 | Warmup 후 Cosine 감소 |
| Warmup Steps | 200 | - |
| Max Steps | 50,000 | - |

---

## 📈 학습 프로세스

### Loss 함수

```python
loss = F.cross_entropy(
    logits[..., :-1, :].reshape(-1, vocab_size),
    labels[..., 1:].reshape(-1),
    ignore_index=pad_token_id,   # PAD만 제외, EOS는 학습
    reduction='mean'
)
```

### 학습 스케줄

```
Step: 0 ──→ 200 (Warmup) ──→ 50,000 (Cosine Decay)
LR  : 0 ──→ 5e-5 ──────────→ ~1e-6
```

### 체크포인트

- 저장 위치: `checkpoints/korean_llm_{step:05d}.pth`
- 저장 내용: step, model_state_dict, optimizer_state_dict, scheduler_state_dict

---

## 🎨 GUI 모니터링

학습 중 자동으로 GUI가 열립니다.

- **왼쪽**: Loss 그래프 (실시간)
- **오른쪽**: 채팅 인터페이스 (CPU에서 최신 체크포인트 로드)

1,000 스텝마다 자동 샘플 생성:

```
Q: 한국의 수도는
A: [모델 생성 답변]

Q: 인공지능이란
A: [모델 생성 답변]

Q: 안녕?
A: [모델 생성 답변]
```

---

## 📝 로깅

```
logs/
├── training.log          # 텍스트 로그
└── loss_history.json     # JSON 손실 기록
```

`loss_history.json` 예시:

```json
[
  {
    "step": 100,
    "loss": 3.6234,
    "lr": 2.5e-5,
    "time": "2024-01-15 10:31:15"
  }
]
```

---

## 📊 성능

### RTX 5090 랩탑 실측

- **1 스텝당 소요 시간**: 약 5초 조금 넘게 걸림

### RTX 4090 기준 예상 (참고)

| 설정 | Tokens/sec | 시간/1K 스텝 | 50K 스텝 예상 |
|------|------------|--------------|---------------|
| batch_size=2, bfloat16 | ~2,000 | ~5분 | ~4시간 |

### 메모리 사용량 (기본 설정)

| 항목 | 메모리 |
|------|--------|
| 모델 가중치 | ~7.2GB |
| 옵티마이저 상태 | ~14.4GB |
| 활성화 (bfloat16) | ~1.0GB |
| 기타 오버헤드 | ~0.4GB |
| **합계** | **~23GB** |

---

## 🐛 트러블슈팅

### CUDA Out of Memory

```python
config = TrainingConfig(
    batch_size=1,
    max_seq_len=128,
    accumulation_steps=16,
)
```

### 데이터셋 다운로드 실패

```bash
rm -rf datasets/cache/*
python korean_llm_advanced_v2.py
```

필요 시 `huggingface-cli login` 실행.

### Loss가 안 줄어들 때

- 학습률을 `1e-4`로 올려보기
- Warmup 스텝을 늘리기
- 데이터셋 다양성 확인

---

## 📚 참고 논문

| 제목 | 연도 | 링크 |
|------|------|------|
| Attention is All You Need | 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| RoPE | 2021 | [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) |
| T5 / RMSNorm | 2020 | [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) |
| SwiGLU | 2020 | [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) |
| Llama | 2023 | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) |
| OpenOrca | 2023 | [arXiv:2310.02959](https://arxiv.org/abs/2310.02959) |

---

## 📄 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다.

사용된 데이터셋 라이선스:
- maywell/korean_textbooks → CC-BY-NC
- squarelike/OpenOrca-gugugo-ko → OpenRAIL
- beomi/KoAlpaca-v1.1a → CC-BY-NC-4.0

각 데이터셋의 라이선스 조건을 확인하고 사용하세요.

---

## 🎯 로드맵

### v2 (현재)
- ✅ 기본 학습 프레임워크
- ✅ GUI 모니터링
- ✅ 데이터셋 통합
- ✅ PAD/EOS 분리

### 예정
- ⏳ LoRA 파인튜닝
- ⏳ Quantization (8-bit / 4-bit)
- ⏳ 분산 학습 (DDP)
- ⏳ Flash Attention V2

---

**마지막 업데이트**: 2026년 8월 8일  
**유지보수자**: seoan1210  
**상태**: 🟢 활발히 유지 중
