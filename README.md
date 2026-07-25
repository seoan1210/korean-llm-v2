# 🇰🇷 Korean LLM v2 - 완전 독자적 한국어 언어모델

![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **RTX 5090에서 처음부터 학습하는 541M 파라미터 한국어 LLM**
>
> 완전히 새로운 구현. 최신 기술만 담았다. 한국어 이해는 기본.

---

## 🎯 프로젝트 소개

한국어 대규모 언어모델(LLM)을 **완전히 처음부터** 개발하는 프로젝트입니다.

### 핵심 특징

- ✅ **완전 독자 개발**: 모든 코드를 직접 작성 (모델, 데이터셋, 학습 엔진)
- ✅ **최신 아키텍처**: RoPE, SwiGLU, Flash Attention, KV-Cache 등 최신 기술 적용
- ✅ **효율적 학습**: Gradient Checkpointing + AMP (bfloat16)로 메모리 절약
- ✅ **빠른 추론**: KV-Cache로 10배 빠른 생성
- ✅ **안정적 학습**: Cosine Annealing + Warmup으로 최적화된 학습 곡선

### 모델 스펙

```
🔧 모델 크기: 541M 파라미터
   - Hidden Size: 1280
   - Layers: 20
   - Attention Heads: 10
   - FFN Hidden: 3200

📊 학습 데이터:
   - maywell/korean_textbooks (한국 교과서)
   - squarelike/OpenOrca-gugugo-ko (OpenOrca 한국어)
   - beomi/KoAlpaca-v1.1a (KoAlpaca 한국어)

⚙️ 하드웨어:
   - GPU: RTX 5090 (24GB VRAM)
   - 배치사이즈: 2 (유효: 64, 누적 32스텝)
   - 학습 속도: 약 0.3초/스텝
```

---

## 🚀 빠른 시작

### 1️⃣ 설치

```bash
# 저장소 클론
git clone https://github.com/seoan1210/korean-llm-v2.git
cd korean-llm-v2

# 필수 패키지 설치
pip install torch transformers datasets

# 추천 (빠른 다운로드)
https://pytorch.org/
사이트에서 맞는 torch 버전 다운로드
```

### 2️⃣ 테스트 실행 (5분)

```bash
python test_korean_llm.py
```

모든 컴포넌트가 제대로 작동하는지 확인합니다:
- ✅ GPU 확인
- ✅ 모델 생성
- ✅ 데이터셋 로드
- ✅ 포워드/역전파
- ✅ 생성 테스트

### 3️⃣ 학습 시작

```bash
python korean_llm_advanced_v2.py
```

기본 설정으로 학습이 시작됩니다. 체크포인트는 10스텝마다 자동 저장됩니다.

### 4️⃣ 커스텀 설정

```python
from korean_llm_advanced_v2 import main, TrainingConfig

config = TrainingConfig(
    batch_size=3,              # 배치사이즈 변경
    accumulation_steps=16,     # 누적 스텝 변경
    max_steps=100000,          # 최대 스텝
    learning_rate=3e-5,        # 학습률
    eval_interval=20,          # 평가 간격
)
main(config)
```

---

## 📁 파일 구조

```
korean-llm-v2/
├── korean_llm_advanced_v2.py   # 메인 학습 코드 (1000줄)
├── test_korean_llm.py           # 테스트 스위트 (8가지 검사)
├── README.md                    # 이 파일
├── QUICK_REFERENCE.md           # 빠른 참조 가이드
└── checkpoints/
    ├── korean_llm_00010.pth     # 10스텝 체크포인트
    ├── korean_llm_00020.pth     # 20스텝 체크포인트
    └── ...
```

---

## 🎯 사용 방법

### 학습된 모델 로드

```python
import torch
from korean_llm_advanced_v2 import KoreanLLM
from transformers import AutoTokenizer

# 모델과 토크나이저 로드
model = KoreanLLM(
    vocab_size=128256,
    pad_token_id=2,
    dim=1280,
    n_layers=20,
    n_heads=10
)
model.load_state_dict(torch.load('korean_llm_00150.pth'))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")

# 생성
from korean_llm_advanced_v2 import generate
response = generate(
    model, tokenizer,
    prompt="한국의 수도는",
    max_tokens=100,
    temperature=0.7,
    device=torch.device("cuda")
)
print(response)
```

### 체크포인트 복구

```python
# 학습 재개하기
checkpoint = torch.load('korean_llm_00150.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

---

## 🔬 기술 사항

### 아키텍처 구성

```
입력 (토큰)
  ↓
임베딩 레이어 (1280차원)
  ↓
Transformer 블록 × 20
  ├─ Multi-Head Attention (10 헤드)
  │  └─ RoPE (Rotary Position Embedding)
  ├─ SwiGLU FFN (3200 히든)
  └─ RMSNorm
  ↓
출력 정규화
  ↓
선형 분류기 (어휘 크기)
  ↓
로짓 (다음 토큰 확률)
```

### 주요 최적화

| 기법 | 효과 | 구현 |
|------|------|------|
| Gradient Checkpointing | 메모리 60% 절약 | 학습 시에만 적용 |
| AMP (bfloat16) | 속도 2배, 메모리 50% | 자동 혼합 정밀도 |
| KV-Cache | 생성 10배 빠름 | 추론 시 키-값 캐싱 |
| Gradient Accumulation | 큰 배치 효과 | 32스텝 누적 |
| Cosine Annealing | 최적 수렴 | Warmup 200스텝 |

### 학습 설정

```python
# 최적화 기본값
Learning Rate:      5e-5
Optimizer:          AdamW
Scheduler:          Cosine Annealing with Warmup
Warmup Steps:       200
Total Steps:        50,000
Grad Clipping:      1.0
```

---

## 🛠️ 문제 해결

### CUDA Out of Memory

```python
# 옵션 1: 배치사이즈 줄이기
config = TrainingConfig(batch_size=1)

# 옵션 2: 시퀀스 길이 줄이기
config = TrainingConfig(max_seq_len=128)

# 옵션 3: 모델 크기 줄이기
dim=768, n_layers=12
```

### 생성이 반복적

```python
# 온도 올리기
response = generate(..., temperature=0.9)

# Top-K 샘플링 조정
response = generate(..., top_k=20)
```

### 데이터셋 다운로드 느림

```bash
# Hugging Face 캐시 위치
~/.cache/huggingface/datasets/

# 캐시 사용 (첫 다운로드 후 자동)
```

---

## 🤝 기여 방법

### 개선 아이디어

이런 것들을 환영합니다:

- 🚀 성능 최적화 (더 빠른 학습)
- 📊 평가 메트릭 (벤치마크)
- 🗣️ 한국어 데이터셋 추가
- 🐛 버그 리포트
- 📝 문서 개선

### 기여 방법

```bash
# 1. Fork
# 2. 브랜치 생성
git checkout -b feature/amazing-feature

# 3. 커밋
git commit -m 'Add amazing feature'

# 4. Push
git push origin feature/amazing-feature

# 5. Pull Request 생성
```

---

## 📚 참고 자료

### 논문

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE
- [Flash Attention](https://arxiv.org/abs/2205.14135) - 효율적 Attention
- [Efficient Transformers](https://arxiv.org/abs/2401.00288) - KV-Cache

### 데이터셋

- [maywell/korean_textbooks](https://huggingface.co/datasets/maywell/korean_textbooks)
- [squarelike/OpenOrca-gugugo-ko](https://huggingface.co/datasets/squarelike/OpenOrca-gugugo-ko)
- [beomi/KoAlpaca-v1.1a](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a)

---

## 📋 로드맵

### Phase 1 (현재) ✅
- [x] 모델 아키텍처 구현
- [x] 학습 엔진 완성

### Phase 2
- [ ] 5000 스텝 학습
- [ ] 한국어 벤치마크 평가
- [ ] 모델 양자화

### Phase 3
- [ ] 모델 크기 증가 (1B+)
- [ ] 분산학습 지원
- [ ] Hugging Face 모델 허브 올리기

### Phase 4
- [ ] LoRA 파인튜닝
- [ ] 특정 도메인 적응
- [ ] API 서빙

---

## ⚖️ 라이센스

MIT License - 자유롭게 사용, 수정, 배포하세요.

자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 👨‍💻 개발자

**만든 사람**: 재기 넘치는 AI 개발자  
**시작 날짜**: 2026-07-25  
**현재 상태**: 활발히 개발 중 🚀

---

## 💬 질문 & 피드백

- 🐛 **버그 리포트**: [Issues](https://github.com/YOUR_USERNAME/korean-llm-v2/issues)
- 💡 **제안**: [Discussions](https://github.com/YOUR_USERNAME/korean-llm-v2/discussions)
- 📧 **이메일**: your.email@example.com

---

## 🙏 감사의 말

감사합니다:

- Hugging Face (데이터셋 & 토크나이저)
- PyTorch 커뮤니티
- 한국 AI 커뮤니티

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요! ⭐**

Made with ❤️ and 🤖

[위로 가기 ⬆️](#-korean-llm-v2---완전-독자적-한국어-언어모델)

</div>
