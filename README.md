🇰🇷 Korean LLM v2 - 완전 독자적 한국어 언어모델

내 컴퓨터에서 처음부터 학습하는 541M 파라미터 한국어 LLM

완전히 새로운 구현. 최신 기술만 담았다. 한국어 이해는 기본.

⸻

🎯 프로젝트 소개

한국어 대규모 언어모델(LLM)을 완전히 처음부터 개발하는 프로젝트입니다.

핵심 특징

* ✅ 완전 독자 개발: 모든 코드를 직접 작성 (모델, 데이터셋, 학습 엔진)
* ✅ 최신 아키텍처: RoPE, SwiGLU, Flash Attention, KV-Cache 등 최신 기술 적용
* ✅ 효율적 학습: Gradient Checkpointing + AMP (bfloat16)로 메모리 절약
* ✅ 빠른 추론: KV-Cache로 빠른 토큰 생성
* ✅ 안정적 학습: Cosine Annealing + Warmup으로 최적화된 학습 곡선

⸻

📊 모델 스펙

🔧 모델 크기: 541M 파라미터
   - Hidden Size: 1280
   - Layers: 20
   - Attention Heads: 10
   - FFN Hidden: 3200
📊 학습 데이터:
   - maywell/korean_textbooks (한국 교과서)
   - squarelike/OpenOrca-gugugo-ko (OpenOrca 한국어)
   - beomi/KoAlpaca-v1.1a (KoAlpaca 한국어)
⚙️ 필요 하드웨어:
   - GPU: VRAM 12GB 이상
   
   지원 시리즈:
   
   ┌─ RTX 50 시리즈 (Blackwell)
   │  ├─ RTX 5090 → 32GB
   │  ├─ RTX 5080 → 16GB
   │  ├─ RTX 5070 Ti → 16GB
   │  ├─ RTX 5070 → 12GB
   │  └─ RTX 5060 Ti → 16GB (8GB 버전도 있음)
   │
   ├─ RTX 40 시리즈 (Ada Lovelace)
   │  ├─ RTX 4090 → 24GB
   │  ├─ RTX 4080 Super → 16GB
   │  ├─ RTX 4080 → 16GB
   │  ├─ RTX 4070 Ti Super → 16GB
   │  ├─ RTX 4070 Ti → 12GB
   │  ├─ RTX 4070 Super → 12GB
   │  ├─ RTX 4070 → 12GB
   │  └─ RTX 4060 Ti → 16GB (8GB 버전도 있음)
   │
   ├─ RTX 30 시리즈 (Ampere)
   │  ├─ RTX 3090 Ti → 24GB
   │  ├─ RTX 3090 → 24GB
   │  ├─ RTX 3080 Ti → 12GB
   │  ├─ RTX 3080 → 12GB (10GB 버전도 있음)
   │  ├─ RTX 3060 → 12GB (8GB 버전도 있음)
   │  └─ TITAN RTX → 24GB
   │
   └─ 노트북 (Laptop / Mobile)
      ├─ RTX 50 시리즈 노트북
      │  ├─ RTX 5090 Laptop → 24GB
      │  ├─ RTX 5080 Laptop → 16GB
      │  ├─ RTX 5070 Ti Laptop → 12GB
      │  └─ RTX 5070 Laptop → 12GB (8GB 버전도 있음)
      │
      ├─ RTX 40 시리즈 노트북
      │  ├─ RTX 4090 Laptop → 16GB
      │  └─ RTX 4080 Laptop → 12GB
      │
      └─ RTX 30 시리즈 노트북
         ├─ RTX 3080 Ti Laptop → 16GB
         ├─ RTX 3080 Laptop → 16GB (일부 모델, 8GB 버전도 있음)
⚙️ 사용된 하드웨어:
   - CPU: Ultra 9 275HX (24c 24t)
   - GPU: RTX 5090 Laptop (24GB GDDR7 VRAM)
   - RAM: 64GB DDR5 6400MHz
   - 배치사이즈: 2 (유효: 64, 누적 32스텝)
   - 학습 속도: 약 0.3초/스텝

⸻

🚀 빠른 시작

1️⃣ 설치

# 저장소 클론
git clone https://github.com/seoan1210/korean-llm-v3.git
cd korean-llm-v3
# 필수 패키지 설치
pip install torch transformers datasets
# (추천) PyTorch 공식 사이트에서 맞는 버전 설치
# https://pytorch.org/

2️⃣ 테스트 실행

python test/test_korean_llm.py

모든 컴포넌트가 제대로 작동하는지 확인합니다:

* ✅ GPU 확인
* ✅ 모델 생성
* ✅ 데이터셋 로드
* ✅ 포워드/역전파
* ✅ 생성 테스트

3️⃣ 학습 시작

python korean_llm_advanced_v2.py

학습을 시작하면 설정된 스텝까지 모델을 학습합니다.

4️⃣ 커스텀 설정

from korean_llm_advanced_v2 import main, TrainingConfig
config = TrainingConfig(
    batch_size=3,
    accumulation_steps=16,
    max_steps=100000,
    learning_rate=3e-5,
    eval_interval=150,
)
main(config)

⸻

📁 파일 구조

KOREAN-LLM-V2/
├── 📂 test/
│   └── test_korean_llm.py           # 테스트 스위트
│
├── korean_llm_advanced_v2.py        # 메인 학습 코드
├── .gitattributes                   # Git LFS 설정
├── LICENSE                          # MIT 라이센스
├── README.md                        # 프로젝트 문서
│
└── 📂 checkpoints/                  # 자동 생성되는 폴더
    ├── korean_llm_00150.pth
    ├── korean_llm_00300.pth
    └── ...

⸻

🎯 사용 방법

기본 학습

python korean_llm_advanced_v2.py

학습 중에는 설정된 체크포인트 간격에 따라 모델 상태가 저장됩니다.

학습된 모델 사용

import torch
from korean_llm_advanced_v2 import KoreanLLM, generate
from transformers import AutoTokenizer
# 체크포인트 로드
checkpoint = torch.load(
    'checkpoints/korean_llm_00150.pth',
    map_location='cuda'
)
model = KoreanLLM(
    vocab_size=128256,
    pad_token_id=2,
    dim=1280,
    n_layers=20,
    n_heads=10
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# 토크나이저
tokenizer = AutoTokenizer.from_pretrained(
    "beomi/Llama-3-Open-Ko-8B"
)
# 생성
response = generate(
    model,
    tokenizer,
    prompt="한국의 수도는",
    max_tokens=100,
    temperature=0.7,
    device=torch.device("cuda")
)
print(response)

⸻

🔬 기술 사항

아키텍처 구성

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

주요 최적화

기법	효과	구현
Gradient Checkpointing	메모리 절약	학습 시 적용
AMP (bfloat16)	속도 및 메모리 효율 향상	자동 혼합 정밀도
KV-Cache	빠른 생성	추론 시 키-값 캐싱
Gradient Accumulation	큰 배치 효과	32스텝 누적
Cosine Annealing	안정적인 수렴	Warmup과 함께 사용

학습 설정

Learning Rate:      5e-5
Optimizer:          AdamW
Scheduler:          Cosine Annealing with Warmup
Warmup Steps:       200
Total Steps:        50,000
Grad Clipping:      1.0
Batch Size:         2
Effective Batch:    64
Accumulation Steps: 32
Max Sequence Length: 256

⸻

🛠️ 문제 해결

CUDA Out of Memory

# 옵션 1: 배치사이즈 줄이기
config = TrainingConfig(batch_size=1)
# 옵션 2: 시퀀스 길이 줄이기
config = TrainingConfig(max_seq_len=128)
# 옵션 3: 모델 크기 줄이기
model = KoreanLLM(
    vocab_size=128256,
    pad_token_id=2,
    dim=768,
    n_layers=12,
    n_heads=10
)

생성이 반복적

# 온도 조절
response = generate(
    ...,
    temperature=0.9
)
# Top-K 샘플링 조정
response = generate(
    ...,
    top_k=20
)

데이터셋 다운로드 느림

Hugging Face 데이터셋 캐시는 기본적으로 로컬에 저장됩니다.
~/.cache/huggingface/datasets/
첫 다운로드 이후에는 캐시를 활용할 수 있습니다.

체크포인트를 찾을 수 없음

import os
print(os.listdir('checkpoints'))

체크포인트 폴더가 없으면 먼저 학습을 시작하고, 설정된 체크포인트 저장 간격에 따라 파일이 생성됩니다.

⸻

📈 학습 곡선 & 진도

Step 0-200:      Warmup
                 학습률을 점진적으로 증가
Step 200-5000:   Main Training
                 본격적인 모델 학습
Step 5000-50000: Cosine Annealing
                 학습률을 점진적으로 감소시키며 수렴

실제 Loss 변화는 데이터 구성, 초기화, 학습률, 배치 구성 및 학습 환경에 따라 달라질 수 있습니다.

⸻

📊 모니터링

학습 상태 확인

# 터미널에서 실시간 로그 확인
tail -f training.log
# 체크포인트 파일 크기 확인
du -h checkpoints/
# 최신 체크포인트 확인
ls -lt checkpoints/ | head -1

저장되는 체크포인트 정보

각 체크포인트에는 다음 정보가 저장됩니다:

* 모델 가중치 (model_state_dict)
* 옵티마이저 상태 (optimizer_state_dict)
* 스케줄러 상태 (scheduler_state_dict)
* 현재 스텝 번호 (step)

⸻

🤝 기여 방법

개선 아이디어

다음과 같은 기여를 환영합니다:

* 🚀 성능 최적화
* 📊 평가 메트릭 및 벤치마크 추가
* 🗣️ 한국어 데이터셋 추가
* 🐛 버그 리포트
* 📝 문서 개선
* 💡 새로운 기능 제안

기여 방법

# 1. Fork
# 2. 브랜치 생성
git checkout -b feature/amazing-feature
# 3. 커밋
git commit -m 'Add amazing feature'
# 4. Push
git push origin feature/amazing-feature
# 5. Pull Request 생성

⸻

📚 참고 자료

논문

* Attention Is All You Need - Transformer
* RoFormer - RoPE
* Flash Attention - 효율적인 Attention
* Efficient Transformers - 효율적인 Transformer 관련 연구

데이터셋

* maywell/korean_textbooks
* squarelike/OpenOrca-gugugo-ko
* beomi/KoAlpaca-v1.1a

⸻

📋 로드맵

Phase 1 — 기본 모델 구현 ✅

* 모델 아키텍처 구현
* Transformer 블록 구현
* RoPE 구현
* SwiGLU 구현
* RMSNorm 구현
* KV-Cache 구현
* 학습 엔진 구현
* 데이터셋 로더 구현

Phase 2 — 학습 및 검증

* 5,000 스텝 이상 학습
* 장시간 학습 안정성 검증
* 한국어 벤치마크 평가
* 생성 품질 평가
* 학습 하이퍼파라미터 최적화

Phase 3 — 모델 개선

* 모델 양자화
* 추론 속도 최적화
* 데이터셋 품질 개선
* 한국어 특화 데이터 추가
* 모델 크기 확장

Phase 4 — 확장

* 1B+ 모델 실험
* 분산학습 지원
* 모델 허브 공개
* 다양한 한국어 태스크 평가

⸻

⚖️ 라이센스

MIT License - 자유롭게 사용, 수정, 배포하세요.

자세한 내용은 LICENSE 파일을 참고하세요.

⸻

👨‍💻 개발자

만든 사람: seoan1210

시작 날짜: 2026-07-25

마지막 업데이트: 2026-07-26

현재 상태: 활발히 개발 중 🚀

⸻

💬 질문 & 피드백

* 🐛 버그 리포트: Issues
* 💬 질문하기: Discussions
* 📧 이메일: seoan1210@example.com

⸻

🙏 감사의 말

감사합니다:

* 🤗 Hugging Face (데이터셋 & 토크나이저)
* 🔥 PyTorch 커뮤니티
* 🇰🇷 한국 AI 커뮤니티
* 👥 모든 기여자들

⸻

🎯 빠른 명령어 치트시트

# 학습 시작
python korean_llm_advanced_v2.py
# 테스트 실행
python test/test_korean_llm.py
# 체크포인트 확인
ls -lh checkpoints/
# 최신 체크포인트 확인
ls -lt checkpoints/ | head -1
# 학습 로그 확인
tail -f training.log

⸻

<div align="center">

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요! ⭐

☕ 커피 한 잔의 후원도 환영합니다!

Made with ❤️ and 🤖

위로 가기 ⬆️

</div>
