🇰🇷 Korean LLM v2 - 완전 독자적 한국어 언어모델

내 컴퓨터에서 처음부터 학습하는 약 541M 파라미터 한국어 LLM

Transformer 아키텍처부터 데이터셋 파이프라인, 학습 루프, 체크포인트, 텍스트 생성까지 직접 구현한 한국어 언어모델 프로젝트입니다.

⸻

🎯 프로젝트 소개

한국어 대규모 언어모델(LLM)을 완전히 처음부터 개발하는 프로젝트입니다.

완성된 LLM을 단순히 가져와 사용하는 것이 아니라, PyTorch를 기반으로 모델의 핵심 구성 요소와 학습 시스템을 직접 구현하고 한국어 데이터로 학습합니다.

핵심 특징

* ✅ 직접 구현: 모델, 데이터셋, 학습 루프, 체크포인트, 생성 함수 구현
* ✅ Transformer 기반 아키텍처: 20개 Transformer Block
* ✅ RoPE: Rotary Position Embedding
* ✅ SwiGLU: 현대적인 Feed-Forward Network
* ✅ RMSNorm: Pre-Norm 기반 정규화
* ✅ Scaled Dot Product Attention: PyTorch 최적화 Attention 사용
* ✅ KV-Cache: autoregressive 생성 시 Key / Value 캐싱
* ✅ Gradient Checkpointing: 학습 시 메모리 절약
* ✅ BF16 AMP: CUDA 환경에서 Mixed Precision 학습
* ✅ Gradient Accumulation: 작은 배치에서 유효 배치 크기 증가
* ✅ Cosine Scheduler + Warmup: 안정적인 학습률 조정
* ✅ Checkpoint 저장/복구: 모델 + Optimizer + Scheduler 상태 저장
* ✅ Top-K + Temperature Sampling: 텍스트 생성 지원

<img width="2752" height="1536" alt="한국어_LLM_아키텍처_학습_분석" src="https://github.com/user-attachments/assets/16fd7b11-b14d-4412-bf2a-ee972e3ed6a0" />

⸻

📊 모델 스펙

🔧 모델 구조
   모델 크기: 약 541M 파라미터
   Hidden Size: 1280
   Layers: 20
   Attention Heads: 10
   Head Dimension: 128
   FFN Hidden: 3200
   Context Length: 256
   Normalization: RMSNorm
   Activation: SwiGLU
   Position Encoding: RoPE
   Attention: Scaled Dot Product Attention
   Weight Tying: 적용
   KV-Cache: 지원
📊 학습 데이터
   - maywell/korean_textbooks
     └─ 한국어 교과서 데이터
   - squarelike/OpenOrca-gugugo-ko
     └─ 한국어 OpenOrca 데이터
   - beomi/KoAlpaca-v1.1a
     └─ 한국어 Instruction 데이터
⚙️ 기본 학습 설정
   Batch Size: 2
   Gradient Accumulation: 32
   Effective Batch Size: 64
   Max Sequence Length: 256
   Learning Rate: 5e-5
   Warmup Steps: 200
   Total Training Steps: 50,000
   Gradient Clipping: 1.0
   Precision: BF16 AMP 지원
💻 사용된 하드웨어
   CPU:
      Intel Core Ultra 9 275HX
      24 Core / 24 Thread
   GPU:
      RTX 5090 Laptop GPU
      24GB GDDR7 VRAM
   RAM:
      64GB DDR5 6400MHz
   Batch Size:
      2
   Gradient Accumulation:
      32
   학습 속도:
      하드웨어 및 환경에 따라 다름

💡 하드웨어 참고

현재 구성은 CUDA GPU 환경을 기준으로 개발되었습니다.

일반적으로 VRAM이 많을수록 큰 Batch Size나 Sequence Length를 사용하기 유리합니다.

다만 실제 필요한 VRAM은 다음 요소에 따라 달라질 수 있습니다.

* GPU 아키텍처
* PyTorch / CUDA 버전
* Batch Size
* Sequence Length
* BF16 사용 여부
* Gradient Checkpointing 여부
* 기타 GPU 메모리 사용량

지원 가능한 GPU 예시

RTX 50 Series
├── RTX 5090
├── RTX 5080
├── RTX 5070 Ti
├── RTX 5070
└── RTX 5060 Ti
RTX 40 Series
├── RTX 4090
├── RTX 4080 SUPER
├── RTX 4080
├── RTX 4070 Ti SUPER
├── RTX 4070 Ti
├── RTX 4070 SUPER
├── RTX 4070
└── RTX 4060 Ti
RTX 30 Series
├── RTX 3090 Ti
├── RTX 3090
├── RTX 3080 Ti
├── RTX 3080
└── RTX 3060
Laptop GPU
├── RTX 5090 Laptop
├── RTX 5080 Laptop
├── RTX 5070 Ti Laptop
├── RTX 5070 Laptop
├── RTX 4090 Laptop
├── RTX 4080 Laptop
└── RTX 3080 Ti / 3080 Laptop

⚠️ 위 목록은 참고용이며 GPU별 VRAM 구성에 따라 실제 실행 가능 여부가 달라질 수 있습니다.

⸻

🚀 빠른 시작

1️⃣ 저장소 클론

git clone https://github.com/seoan1210/korean-llm-v2.git
cd korean-llm-v2

2️⃣ 필수 패키지 설치

pip install torch transformers datasets

GPU를 사용하는 경우 자신의 CUDA 환경에 맞는 PyTorch 버전을 설치하는 것을 권장합니다.

PyTorch 공식 설치 페이지:

https://pytorch.org/

⸻

3️⃣ 테스트 실행

테스트 파일이 저장된 위치에 따라 실행합니다.

python test/test_korean_llm.py

테스트에서는 다음과 같은 주요 기능을 확인할 수 있습니다.

* ✅ GPU 확인
* ✅ 모델 생성
* ✅ 데이터셋 로드
* ✅ Forward Pass
* ✅ Backward Pass
* ✅ 텍스트 생성
* ✅ 주요 컴포넌트 동작 확인

⸻

4️⃣ 학습 시작

python korean_llm_advanced_v2.py

기본 설정으로 학습이 시작됩니다.

학습 과정에서 설정된 간격에 따라 평가 샘플을 생성하고 체크포인트를 저장합니다.

⸻

⚙️ 커스텀 설정

TrainingConfig를 이용하면 학습 환경을 직접 변경할 수 있습니다.

from korean_llm_advanced_v2 import main, TrainingConfig
config = TrainingConfig(
    batch_size=3,
    accumulation_steps=16,
    max_steps=100000,
    learning_rate=3e-5,
    eval_interval=150,
)
main(config)

주요 설정:

batch_size
    GPU에 한 번에 넣는 데이터 개수
accumulation_steps
    Gradient Accumulation 횟수
max_steps
    최대 학습 반복 횟수
learning_rate
    학습률
warmup_steps
    Warmup 단계
checkpoint_interval
    체크포인트 저장 간격 설정
eval_interval
    평가 및 샘플 생성 간격
max_seq_len
    최대 입력 토큰 길이
use_bfloat16
    BF16 AMP 사용 여부
resume_from_checkpoint
    체크포인트 복구 경로 또는 'latest'

⸻

📁 파일 구조

korean-llm-v2/
│
├── korean_llm_advanced_v2.py
│   └── 모델 / 데이터셋 / 학습 / 생성 / 체크포인트
│
├── test/
│   └── test_korean_llm.py
│       └── 테스트 스위트
│
├── README.md
│
├── QUICK_REFERENCE.md
│
└── checkpoints/
    ├── korean_llm_00100.pth
    ├── korean_llm_00200.pth
    ├── korean_llm_00300.pth
    └── ...

⸻

🎯 사용 방법

학습된 모델 로드

체크포인트는 모델 가중치만 저장하는 것이 아니라 학습 상태 전체를 저장합니다.

import torch
from korean_llm_advanced_v2 import KoreanLLM, generate
from transformers import AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(
    "checkpoints/korean_llm_00100.pth",
    map_location=device
)
model = KoreanLLM(
    vocab_size=128256,
    pad_token_id=2,
    dim=1280,
    n_layers=20,
    n_heads=10,
    max_seq_len=256
).to(device)
model.load_state_dict(
    checkpoint["model_state_dict"]
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    "beomi/Llama-3-Open-Ko-8B"
)
tokenizer.pad_token = tokenizer.eos_token
response = generate(
    model,
    tokenizer,
    prompt="한국의 수도는",
    max_tokens=100,
    temperature=0.7,
    top_k=40,
    device=device
)
print(response)

⚠️ vocab_size와 pad_token_id는 실제 사용한 tokenizer 설정과 일치해야 합니다.

⸻

🔄 체크포인트 복구

학습을 중단했다가 다시 이어서 학습할 수 있습니다.

가장 최근 체크포인트를 자동으로 찾으려면:

from korean_llm_advanced_v2 import main, TrainingConfig
config = TrainingConfig(
    resume_from_checkpoint="latest"
)
main(config)

특정 체크포인트를 지정할 수도 있습니다.

config = TrainingConfig(
    resume_from_checkpoint=
        "checkpoints/korean_llm_00100.pth"
)
main(config)

체크포인트에는 다음 정보가 포함됩니다.

Checkpoint
├── step
├── model_state_dict
├── optimizer_state_dict
└── scheduler_state_dict

따라서 모델 가중치뿐 아니라 Optimizer와 Scheduler의 상태도 함께 복구할 수 있습니다.

⸻

🔬 기술 사항

아키텍처 구성

입력 (Token IDs)
        │
        ▼
Token Embedding
   1280 Dimensions
        │
        ▼
┌───────────────────────────┐
│   Transformer Block ×20   │
│                           │
│   RMSNorm                 │
│      ↓                    │
│   Multi-Head Attention    │
│      ↓                    │
│   RoPE                    │
│      ↓                    │
│   Residual Connection     │
│                           │
│   RMSNorm                 │
│      ↓                    │
│   SwiGLU FFN              │
│      ↓                    │
│   Residual Connection     │
└───────────────────────────┘
        │
        ▼
      RMSNorm
        │
        ▼
Output Projection
        │
        ▼
      Logits
        │
        ▼
Next Token Prediction

⸻

🧠 RMSNorm

Transformer의 정규화에는 RMSNorm을 사용합니다.

class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        ) * self.weight

각 Transformer Block에서 Pre-Norm 방식으로 사용됩니다.

⸻

🔄 RoPE

위치 정보를 표현하기 위해 Rotary Position Embedding을 사용합니다.

각 Attention Head의 차원은:

1280 / 10 = 128

이며 이 Head Dimension에 RoPE가 적용됩니다.

Query ──┐
        ├── RoPE ── Attention
Key ────┘

⸻

🔥 SwiGLU

FFN에는 SwiGLU 구조를 사용합니다.

             Input
                │
        ┌───────┴───────┐
        ▼               ▼
      Linear          Linear
        │               │
       SiLU             │
        │               │
        └────── × ──────┘
                │
                ▼
             Linear
                │
                ▼
              Output

구현:

return self.w2(
    F.silu(self.w1(x)) * self.w3(x)
)

FFN Hidden Dimension은 3200입니다.

⸻

⚡ Attention

PyTorch의 scaled_dot_product_attention()을 사용합니다.

F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=mask,
    is_causal=(mask is None and s > 1)
)

이를 통해 PyTorch가 제공하는 최적화된 Scaled Dot Product Attention 경로를 활용합니다.

⸻

💾 KV-Cache

추론 시 이전 Key / Value를 캐시하여 autoregressive generation을 효율적으로 처리합니다.

첫 번째 입력
      │
      ▼
  K / V 계산
      │
      ▼
   KV Cache
      │
      ▼
새로운 Token
      │
      ▼
새로운 K / V
      │
      ▼
기존 Cache와 결합

코드에서는 다음과 같이 이전 캐시와 새로운 Key / Value를 연결합니다.

k = torch.cat([pk, k], dim=2)
v = torch.cat([pv, v], dim=2)

⸻

🧠 Gradient Checkpointing

학습 중 GPU 메모리 사용량을 줄이기 위해 Transformer Layer에 Gradient Checkpointing을 적용합니다.

x, kv = checkpoint(
    layer,
    x,
    f_cos,
    f_sin,
    None,
    use_reentrant=False
)

중간 activation을 모두 저장하는 대신 backward 과정에서 일부 연산을 다시 수행하여 메모리와 연산량 사이의 균형을 맞춥니다.

⸻

⚡ BF16 AMP

CUDA 환경에서는 BF16 Mixed Precision 학습을 지원합니다.

with torch.amp.autocast(
    "cuda",
    dtype=torch.bfloat16
):
    _, loss, _ = model(
        batch,
        labels=batch
    )

TrainingConfig에서 활성화 여부를 변경할 수 있습니다.

config = TrainingConfig(
    use_bfloat16=True
)

⸻

📦 Gradient Accumulation

기본 설정:

Batch Size
    2
Accumulation Steps
    32
Effective Batch Size
    2 × 32 = 64

즉, 매 micro-batch마다 optimizer를 업데이트하지 않고 gradient를 누적한 뒤 업데이트합니다.

⸻

📈 Optimizer & Scheduler

Optimizer:

AdamW

기본 Learning Rate:

5e-5

Scheduler:

Cosine Schedule with Warmup

기본 설정:

Learning Rate:       5e-5
Warmup Steps:        200
Training Steps:      50,000
Gradient Clipping:   1.0

⸻

💡 Weight Tying

입력 Embedding과 출력 Projection의 가중치를 공유합니다.

self.output.weight = self.embed.weight

이를 통해 두 레이어가 동일한 weight parameter를 공유하도록 구성했습니다.

⸻

📊 Loss

언어모델 학습에는 다음 토큰을 예측하는 Causal Language Modeling 방식을 사용합니다.

입력:
나는 오늘 학교에
예측:
나는 → 오늘
오늘 → 학교
학교 → 에

코드에서는 입력과 label을 한 칸씩 이동시켜 Cross Entropy Loss를 계산합니다.

loss = F.cross_entropy(
    logits[..., :-1, :].reshape(-1, logits.size(-1)),
    labels[..., 1:].reshape(-1),
    ignore_index=self.pad_token_id
)

PAD 토큰은 Loss 계산에서 제외됩니다.

⸻

🎲 텍스트 생성

텍스트 생성은 다음과 같이 사용할 수 있습니다.

response = generate(
    model,
    tokenizer,
    prompt="인공지능이란",
    max_tokens=100,
    temperature=0.7,
    top_k=40,
    device=device
)
print(response)

생성 과정:

Prompt
  ↓
Tokenizer
  ↓
Transformer
  ↓
KV-Cache
  ↓
Temperature
  ↓
Top-K Filtering
  ↓
Sampling
  ↓
Next Token
  ↓
반복

⸻

🌡️ Temperature

Temperature는 다음 토큰의 선택 분포를 조절합니다.

temperature=0.7

낮은 값은 비교적 높은 확률의 토큰을 선택하는 경향을 만들고, 높은 값은 더 다양한 토큰 선택을 허용합니다.

예:

response = generate(
    model,
    tokenizer,
    prompt="한국의 수도는",
    temperature=0.9
)

⸻

🎯 Top-K Sampling

기본 설정은:

top_k=40

입니다.

전체 Vocabulary에서 확률이 높은 상위 K개의 토큰만 남긴 뒤 샘플링합니다.

전체 Vocabulary
       │
       ▼
   Top-K Filter
       │
       ▼
  상위 40개 Token
       │
       ▼
  Temperature
       │
       ▼
   Multinomial
       │
       ▼
   Next Token

⸻

🛠️ 문제 해결

CUDA Out of Memory

옵션 1 — Batch Size 감소

config = TrainingConfig(
    batch_size=1
)

옵션 2 — Sequence Length 감소

config = TrainingConfig(
    max_seq_len=128
)

옵션 3 — Gradient Accumulation 조정

config = TrainingConfig(
    batch_size=1,
    accumulation_steps=64
)

작은 batch를 사용하면서 gradient accumulation 횟수를 늘려 유효 배치 크기를 유지할 수 있습니다.

⸻

BF16을 사용할 수 없는 GPU

config = TrainingConfig(
    use_bfloat16=False
)

⸻

생성 결과가 반복되는 경우

Temperature를 조절할 수 있습니다.

response = generate(
    model,
    tokenizer,
    prompt="인공지능이란",
    temperature=0.9
)

Top-K도 변경할 수 있습니다.

response = generate(
    model,
    tokenizer,
    prompt="인공지능이란",
    top_k=20
)

⸻

데이터셋 다운로드가 느린 경우

Hugging Face Dataset은 로컬 캐시를 사용할 수 있습니다.

~/.cache/huggingface/datasets/

첫 실행 이후 환경에 따라 캐시된 데이터가 재사용될 수 있습니다.

⸻

📈 학습 모니터링

학습 과정에서는 다음과 같은 정보가 출력됩니다.

[Step     1] Loss: ... | LR: ... | Tokens/step: ...
[Step     2] Loss: ... | LR: ... | Tokens/step: ...
...

또한 평가 간격마다 다음과 같은 prompt로 생성 결과를 확인합니다.

한국의 수도는
인공지능이란
좋은 날씨에는

이를 통해 학습 도중 모델의 생성 결과가 어떻게 변화하는지 확인할 수 있습니다.

⸻

💾 체크포인트

체크포인트는 다음과 같은 형태로 저장됩니다.

checkpoints/
├── korean_llm_00100.pth
├── korean_llm_00200.pth
├── korean_llm_00300.pth
└── ...

저장되는 정보:

Model
├── model_state_dict
Optimizer
├── optimizer_state_dict
Scheduler
├── scheduler_state_dict
Training
└── step

학습 중 오류나 중단이 발생하더라도 저장된 체크포인트를 이용하여 학습 상태를 복구할 수 있습니다.

⸻

🤝 기여 방법

개선 아이디어

다음과 같은 기여를 환영합니다.

* 🚀 학습 성능 최적화
* 📊 한국어 평가 벤치마크 추가
* 🗣️ 한국어 데이터셋 추가
* 🐛 버그 리포트
* 📝 문서 개선
* 💡 새로운 기능 제안
* ⚡ 추론 최적화

기여 방법

# 1. Fork
# 2. 브랜치 생성
git checkout -b feature/amazing-feature
# 3. 수정 후 Commit
git commit -m "Add amazing feature"
# 4. Push
git push origin feature/amazing-feature
# 5. Pull Request 생성

⸻

📚 참고 자료

논문

* Attention Is All You Need - Transformer
* RoFormer - RoPE
* FlashAttention - 효율적인 Attention
* Efficient Transformers - Transformer 효율화 연구

데이터셋

* maywell/korean_textbooks
* squarelike/OpenOrca-gugugo-ko
* beomi/KoAlpaca-v1.1a

Tokenizer

* beomi/Llama-3-Open-Ko-8B

⸻

📋 로드맵

Phase 1 — 기본 구현 ✅

* Transformer 아키텍처 구현
* RMSNorm 구현
* RoPE 구현
* SwiGLU 구현
* Multi-Head Attention 구현
* Scaled Dot Product Attention 적용
* KV-Cache 구현
* Weight Tying
* Gradient Checkpointing
* BF16 AMP
* Gradient Accumulation
* Cosine Scheduler + Warmup
* Checkpoint 저장 / 복구
* 한국어 데이터셋 Streaming

Phase 2 — 학습 및 평가 🚧

* 5,000 스텝 이상 학습
* 장시간 학습 안정성 검증
* 한국어 벤치마크 평가
* 생성 품질 평가
* 학습 하이퍼파라미터 최적화

Phase 3 — 모델 개선 📋

* 데이터 품질 개선
* 한국어 데이터셋 확장
* 추론 속도 최적화
* 모델 양자화
* 학습 안정성 개선

Phase 4 — 모델 확장 📋

* 모델 크기 증가
* 1B+ 모델 실험
* 분산학습 지원
* Hugging Face 모델 허브 공개
* API 서빙

Phase 5 — 추가 학습 📋

* LoRA 파인튜닝
* 특정 도메인 적응
* Instruction Tuning
* 대화형 모델 실험

⸻

⚖️ 라이센스

MIT License

자유롭게 사용, 수정, 배포하세요.

자세한 내용은 LICENSE 파일을 참고하세요.

⸻

👨‍💻 개발자

만든 사람: seoan1210

시작 날짜: 2026-07-25

현재 상태: 활발히 개발 중 🚀

📊 프로젝트 자료

슬라이드쇼:
Korean-LLM-V2_Deep_Dive.pptx

⸻

💬 질문 & 피드백

* 🐛 버그 리포트: Issues

⸻

🙏 감사의 말

감사합니다.

* 🤗 Hugging Face — 데이터셋 및 토크나이저
* 🔥 PyTorch 커뮤니티
* 🇰🇷 한국 AI 커뮤니티
* 👥 오픈소스 생태계의 모든 기여자

⸻

<div align="center">

🇰🇷 Korean LLM V2

From Architecture to Training — Built with PyTorch

직접 설계하고, 직접 구현하고, 직접 학습하는 한국어 LLM

⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐

Made with ❤️ and 🤖

⬆️ 위로 가기

</div>
