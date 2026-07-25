# 🇰🇷 Korean LLM v2 - 완전 독자적 한국어 언어모델

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **내 컴퓨터에서 처음부터 학습하는 541M 파라미터 한국어 LLM**
>
> 완전히 새로운 구현. 최신 Transformer 기술을 적용한 독자 한국어 언어모델.

---

# 🎯 프로젝트 소개

한국어 대규모 언어모델(LLM)을 **처음부터 직접 구현하고 학습하는 프로젝트**입니다.

모델 구조, 데이터 파이프라인, 학습 엔진, 추론 시스템까지 직접 개발합니다.

## 핵심 특징

- ✅ 완전 독자 개발
  - 모델 구조 직접 구현
  - 학습 루프 직접 작성
  - 데이터 처리 파이프라인 구축

- ✅ 최신 Transformer 기술 적용
  - RoPE
  - SwiGLU
  - Flash Attention
  - KV Cache

- ✅ 효율적인 학습
  - Gradient Checkpointing
  - AMP (bfloat16)
  - Gradient Accumulation

- ✅ 빠른 추론
  - KV Cache 기반 생성
  - 효율적인 메모리 사용

- ✅ 안정적인 학습
  - Warmup
  - Cosine Scheduler
  - AdamW Optimizer

---

# 🧠 모델 스펙

```
Parameters:
541M

Hidden Size:
1280

Layers:
20

Attention Heads:
10

FFN Hidden:
3200

Max Sequence Length:
256

Vocabulary Size:
128,257
```

---

# 📚 학습 데이터

사용 데이터셋:

- `maywell/korean_textbooks`
- `squarelike/OpenOrca-gugugo-ko`
- `beomi/KoAlpaca-v1.1a`

---

# ⚙️ 하드웨어

## 테스트 환경

```
CPU:
Ultra 9 275HX
24 Core / 24 Thread

GPU:
RTX 5090 Laptop
24GB GDDR7 VRAM

RAM:
64GB DDR5 6400MHz
```

학습 설정:

```
Batch Size:
2

Gradient Accumulation:
32

Effective Batch Size:
64

Speed:
약 0.3초 / step
```

---

# 🚀 빠른 시작

## 설치

```bash
git clone https://github.com/seoan1210/korean-llm-v2.git

cd korean-llm-v2

pip install torch transformers datasets
```

---

# 테스트

```bash
python test_korean_llm.py
```

검사 항목:

- GPU 확인
- 모델 생성
- 데이터셋 로드
- Forward / Backward
- 생성 테스트

---

# 학습 시작

```bash
python korean_llm_advanced_v2.py
```

체크포인트:

```
checkpoints/

├── korean_llm_00100.pth
├── korean_llm_00200.pth
└── ...
```

---

# 🔧 커스텀 설정

```python
from korean_llm_advanced_v2 import main, TrainingConfig

config = TrainingConfig(
    batch_size=3,
    accumulation_steps=16,
    max_steps=100000,
    learning_rate=3e-5
)

main(config)
```

---

# 📁 프로젝트 구조

```
korean-llm-v2/

├── korean_llm_advanced_v2.py
├── test_korean_llm.py
├── README.md
├── QUICK_REFERENCE.md

└── checkpoints/

    ├── korean_llm_00100.pth
    ├── korean_llm_00200.pth
    └── ...
```

---

# 🏗️ 모델 구조

```
Input Token

↓

Embedding
1280 Dimension

↓

Transformer Block × 20

 ├─ Multi Head Attention
 │
 ├─ RoPE
 │
 ├─ SwiGLU FFN
 │
 └─ RMSNorm

↓

Linear Output

↓

Next Token Prediction
```

---

# ⚡ 최적화 기술

| 기술 | 효과 |
|---|---|
| Gradient Checkpointing | 메모리 절약 |
| AMP bfloat16 | 학습 속도 향상 |
| KV Cache | 빠른 생성 |
| Gradient Accumulation | 큰 배치 효과 |
| Cosine Scheduler | 안정적 수렴 |

---

# 📈 학습 설정

```
Optimizer:
AdamW

Learning Rate:
5e-5

Scheduler:
Cosine Annealing + Warmup

Warmup Steps:
200

Gradient Clip:
1.0

Total Steps:
5000+
```

---

# 🛠️ 문제 해결

## CUDA Out Of Memory

배치 감소:

```python
TrainingConfig(
    batch_size=1
)
```

시퀀스 감소:

```python
TrainingConfig(
    max_seq_len=128
)
```

모델 축소:

```python
dim=768
n_layers=12
```

---

# 🗺️ 로드맵

## Phase 1 ✅

- [x] 모델 아키텍처 구현
- [x] 학습 엔진 구현


## Phase 2

- [ ] 5000 Step 학습
- [ ] 한국어 벤치마크 평가
- [ ] 모델 양자화


## Phase 3

- [ ] 1B+ 모델
- [ ] 분산 학습
- [ ] Hugging Face 공개


## Phase 4

- [ ] LoRA Fine-tuning
- [ ] 도메인 특화
- [ ] API 서버

---

# ⚖️ License

MIT License

자유롭게 사용, 수정, 배포 가능합니다.

---

# 👨‍💻 Developer

**seoan1210**

시작 날짜:
2026-07-25

상태:
Active Development 🚀

---

# 🙏 Thanks

- Hugging Face
- PyTorch Community
- Korean AI Community


⭐ 프로젝트가 도움이 되었다면 Star 부탁드립니다 ⭐

Made with ❤️ and 🤖