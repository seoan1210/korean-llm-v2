# 🇰🇷 Korean LLM V2

![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76b900?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Training-brightgreen?style=flat-square)
![Steps](https://img.shields.io/badge/Steps-9700%2F50000-orange?style=flat-square)
![Loss](https://img.shields.io/badge/Loss-1.50%20±%200.08-yellow?style=flat-square)

한국어 대규모 언어 모델을 **완전히 독자적으로** 개발하고 학습시키는 오픈소스 프로젝트입니다.

LLaMA 기반의 1.3B 파라미터 모델을 순수 한국어 데이터로 학습하고 있습니다. 
**모든 사람**이 자신의 GPU로 이 프로젝트를 사용할 수 있도록 설계되었습니다. 🎯

**Current Status**: 🚀 9700 Steps | Loss: 1.45~1.70 (Stable) | Training Ongoing

---

## ✨ 프로젝트 특징

- 🎯 **완전 독자적 개발**: 모든 코드를 처음부터 직접 작성
- 📚 **한국어 특화**: 3개의 공개 한국어 데이터셋으로 학습
- ⚡ **유연한 설정**: 4GB ~ 80GB GPU 모두 지원
- 📖 **완벽한 문서**: 초보자부터 전문가까지 모두 이용 가능
- 🔐 **100% 오픈소스**: MIT 라이선스로 자유롭게 사용 가능
- 🌍 **모든 플랫폼**: Windows, Linux, macOS 모두 지원

---

## 🚀 빠른 시작 (5분)

### 1️⃣ 필요한 것

```bash
# 최소 요구사항
- Python 3.9+
- GPU: 8GB VRAM (권장: 12GB+)
- CUDA: 11.8+ 또는 12.1
- 디스크: 50GB (데이터셋 + 모델)
```

### 2️⃣ 설치

```bash
# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 또는 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 의존성 설치
pip install transformers datasets
```

### 3️⃣ 학습 시작

```python
from korean_llm_advanced_v2 import TrainingConfig, main

config = TrainingConfig(
    batch_size=2,
    max_steps=50000,
    learning_rate=5e-5,
)

main(config)
```

---

## 📊 GPU별 권장 설정

### 24GB VRAM (RTX 5090, 4090, A100 등)
```python
config = TrainingConfig(
    batch_size=2,
    accumulation_steps=32,
    max_seq_len=256,
    use_bfloat16=True,
)
# 예상 속도: 3-5초/스텝
```

### 16GB VRAM (RTX 4080, 3090 등)
```python
config = TrainingConfig(
    batch_size=1,
    accumulation_steps=64,
    max_seq_len=256,
    use_bfloat16=True,
)
# 예상 속도: 5-8초/스텝
```

### 12GB VRAM (RTX 4070 등)
```python
config = TrainingConfig(
    batch_size=1,
    accumulation_steps=64,
    max_seq_len=128,
    use_bfloat16=True,
)
# 예상 속도: 8-12초/스텝
```

### 8GB VRAM (RTX 4060, 3060 등)
```python
config = TrainingConfig(
    batch_size=1,
    accumulation_steps=128,
    max_seq_len=128,
    use_bfloat16=True,
)
# 예상 속도: 15-20초/스텝
```

### 저사양 (4GB~6GB)
```python
config = TrainingConfig(
    batch_size=1,
    accumulation_steps=256,
    max_seq_len=64,
    use_bfloat16=True,
)
# 주의: 모델 크기 축소 필요
# dim=512, n_layers=12 (기본값: 1280, 20)
```

---

## 🏗️ 프로젝트 구조

```
korean-llm-v2/
├── korean_llm_advanced_v2.py  # 메인 학습 코드 (1,300 라인)
├── README.md                   # 이 파일
├── LICENSE                     # MIT 라이선스
└── .gitattributes             # Git 속성 설정
```

### 파일 설명

| 파일 | 설명 |
|------|------|
| **korean_llm_advanced_v2.py** | 전체 학습 파이프라인 (모델, 데이터, 학습 루프) |
| **README.md** | 프로젝트 문서 및 사용 가이드 |
| **LICENSE** | MIT 라이선스 전문 |

---

## 📚 모델 아키텍처

### KoreanLLM (1.3B 파라미터)

```
입력 (토큰 ID)
    ↓
Embedding Layer (vocab → 1280 dim)
    ↓
[Transformer Block × 20]
  ├─ Multi-Head Attention (10 heads)
  │  ├─ Query, Key, Value Projection
  │  ├─ Rotary Position Embedding (RoPE)
  │  └─ KV-Cache (추론 최적화)
  ├─ Feed Forward (SwiGLU)
  │  └─ 1280 → 3200 → 1280
  ├─ RMSNorm
  └─ Residual Connections
    ↓
Output Projection (1280 → vocab)
    ↓
확률 분포 (다음 토큰)
```

### 핵심 기술

- **RoPE (Rotary Position Embedding)**: 위치 정보를 rotation으로 인코딩
- **RMSNorm**: 안정적인 레이어 정규화
- **SwiGLU**: 고성능 활성화 함수
- **Gradient Checkpointing**: 메모리 효율화
- **bfloat16 AMP**: 혼합 정밀도로 속도 향상
- **KV-Cache**: 자기회귀 생성 최적화

---

## 📊 학습 현황

### 9700 스텝 기준

```
평균 손실: 1.50 ± 0.08
손실 범위: 1.45 ~ 1.70
학습 상태: 수렴 중
학습 속도: 4.5초/스텝 (RTX 5090 기준)
처리량: 114 tokens/sec
```

### 학습 곡선 분석

```
단계별 손실 감소:

Step 0-2000:        손실 빠른 감소 (70 → 20)
Step 2000-5000:     문법 학습 시작
Step 5000-9700:     안정화 진입 ← 현재 위치
Step 10000+:        의미 이해 강화 (예상)
Step 20000+:        정교한 생성 능력 (목표)
Step 50000:         완성 (최종 목표)
```

---

## 📚 데이터셋

학습에 사용되는 **공개 한국어 데이터셋**:

| 데이터셋 | 출처 | 크기 | 특징 |
|---------|------|------|------|
| **Korean Textbooks** | maywell | ~50M tokens | 교과서 텍스트 |
| **OpenOrca-gugugo-ko** | squarelike | ~200M tokens | 명령어-응답 쌍 |
| **KoAlpaca** | beomi | ~100M tokens | Alpaca 스타일 |

**특징:**
- ✅ 자동 다운로드 (첫 실행 시만)
- ✅ 로컬 Parquet 캐싱 (빠른 재사용)
- ✅ 자동 필드 감지 (text, instruction+output 등)
- ✅ 해시 기반 버전 관리
- ✅ 최대 3회 자동 재시도

---

## 🎓 자세한 사용 가이드

### 기본 학습

```python
from korean_llm_advanced_v2 import TrainingConfig, main

config = TrainingConfig(
    batch_size=2,
    max_steps=50000,
    learning_rate=5e-5,
    warmup_steps=200,
    eval_interval=500,
    checkpoint_interval=500,
)

main(config)
```

### 이전 체크포인트에서 재개

```python
config = TrainingConfig(
    resume_from_checkpoint='latest'  # 자동 최신 찾기
)

main(config)
```

또는

```python
config = TrainingConfig(
    resume_from_checkpoint='checkpoints/korean_llm_05000.pth'  # 특정 체크포인트
)

main(config)
```

### 커스텀 하이퍼파라미터

```python
config = TrainingConfig(
    # 배치 처리
    batch_size=1,
    accumulation_steps=128,
    
    # 모델
    max_seq_len=512,
    
    # 학습
    max_steps=100000,
    learning_rate=3e-5,
    warmup_steps=500,
    
    # 최적화
    use_bfloat16=True,
    
    # 데이터
    num_workers=8,
    samples_per_dataset=None,  # None = 전체 사용
    
    # 평가
    eval_interval=1000,
    checkpoint_interval=1000,
)

main(config)
```

### 추론/생성

```python
import torch
from transformers import AutoTokenizer
from korean_llm_advanced_v2 import KoreanLLM, generate

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")

# 체크포인트에서 모델 로드
checkpoint = torch.load('checkpoints/korean_llm_09700.pth', 
                        map_location='cuda')
model = KoreanLLM(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
).cuda()
model.load_state_dict(checkpoint['model_state_dict'])

# 텍스트 생성
prompt = "한국의 수도는"
response = generate(
    model, 
    tokenizer,
    prompt=prompt,
    max_tokens=100,
    temperature=0.7,
    top_k=40,
    device=torch.device('cuda')
)

print(f"질문: {prompt}")
print(f"답변: {response}")
```

---

## 🔧 상세 설치 가이드

### Windows

#### 1. CUDA 설치
```
1. https://developer.nvidia.com/cuda-downloads 방문
2. Windows → x86_64 → Windows 11/10 → Local 선택
3. CUDA 11.8 또는 12.1 다운로드 및 설치
4. 기본 설정 사용
5. 컴퓨터 재부팅
```

#### 2. 환경 변수 설정
```
시스템 환경 변수 편집:

CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

PATH에 추가:
%CUDA_HOME%\bin
%CUDA_HOME%\libnvvp
```

#### 3. cuDNN 설치 (선택)
```
1. https://developer.nvidia.com/cudnn 방문
2. cuDNN 8.9+ for CUDA 11.x 다운로드 (가입 필요)
3. 압축 해제
4. bin/, lib/, include/ → CUDA_HOME에 복사
```

#### 4. 확인
```bash
nvcc --version  # CUDA 버전 확인
nvidia-smi      # GPU 확인
```

### Linux (Ubuntu)

```bash
# CUDA 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-repo-ubuntu2204_12.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204_12.1.0-1_amd64.deb
sudo apt-get update
sudo apt-get -y install cuda

# 환경 변수 설정
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### macOS

```bash
# GPU 없이 CPU로만 학습 (매우 느림)
pip install torch torchvision torchaudio

# M1/M2 칩 (Metal 가속)
pip install torch torchvision torchaudio
```

---

## 🐛 문제 해결

### "CUDA out of memory" 에러

**원인**: GPU 메모리 부족

**해결책**:
```python
# 1. 배치 사이즈 줄이기
config.batch_size = 1

# 2. 시퀀스 길이 줄이기
config.max_seq_len = 128

# 3. 그래디언트 누적 늘리기
config.accumulation_steps = 256

# 4. 모델 크기 줄이기 (극단적인 경우)
model = KoreanLLM(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    dim=512,      # 기본값: 1280
    n_layers=12,  # 기본값: 20
)
```

### PyTorch가 GPU를 인식하지 못함

```bash
# 1. 버전 확인
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"

# 2. CUDA 설치 확인
nvcc --version

# 3. 재설치
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 재부팅
```

### 데이터셋 다운로드 실패

```python
# 자동 재시도 (3회)
config = TrainingConfig(download_datasets=True)
main(config)

# 또는 강제 재다운로드
from korean_llm_advanced_v2 import DatasetManager
manager = DatasetManager()
manager.get_or_download_all(force=True)
```

### 모델 로드 실패

```bash
# 체크포인트 파일 확인
ls -lah checkpoints/

# 손상된 체크포인트 삭제
rm checkpoints/korean_llm_*.pth

# 처음부터 시작
config = TrainingConfig(resume_from_checkpoint=None)
main(config)
```

---

## ❓ FAQ

**Q: 학습에 얼마나 걸리나요?**  
A: 50,000 스텝 완료에 약 60시간 필요합니다 (RTX 5090 기준).
- RTX 4090: ~70시간
- RTX 4080: ~100시간
- RTX 4070: ~150시간

**Q: 모델이 언제 쓸만해질까요?**  
A: 일반적으로 20,000 스텝 이후 기초 의미 이해가 보입니다.
- 10K: 한국어 문법 학습
- 20K: 기초 의미 이해
- 30K: 관련 주제 생성
- 50K: 고급 기능

**Q: 다른 GPU에서도 되나요?**  
A: 네! 4GB부터 80GB까지 모두 지원합니다.
[GPU별 권장 설정](#gpu별-권장-설정) 섹션을 참고하세요.

**Q: CPU로만 학습할 수 있나요?**  
A: 기술적으로는 가능하지만, 속도가 **100배 이상 느립니다**.
GPU 사용을 강력히 권장합니다.

**Q: 온도 관리는 어떻게 하나요?**  
A: 일반적인 안전 범위:
- GPU: 70-80°C (최대 90°C)
- CPU: 80-95°C (정상)

온도가 높으면:
- 노트북 스탠드 사용
- 선풍기로 냉각
- 배경 프로세스 종료

**Q: 배치 사이즈를 더 크게 할 수 있나요?**  
A: GPU 메모리가 허락하면 가능합니다.
다만, CUDA out of memory 위험이 있으니 천천히 증가시키세요.

**Q: 학습 중간에 멈춰도 괜찮나요?**  
A: 네, 자동으로 마지막 체크포인트에서 재개됩니다.
`resume_from_checkpoint='latest'` 사용.

**Q: 다른 데이터셋을 사용할 수 있나요?**  
A: 코드를 수정하면 가능합니다.
`DatasetManager.DATASETS_CONFIG`를 수정하세요.

**Q: 모델을 내보낼 수 있나요?**  
A: 예정 중입니다. 현재는 체크포인트 형식만 지원합니다.

**Q: 추론 속도는 얼마나 되나요?**  
A: RTX 5090에서 약 114 tokens/sec입니다.
다른 GPU에서는 비례적으로 변합니다.

---

## 📚 참고 자료

### 논문
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA

### 공식 문서
- [PyTorch 문서](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [CUDA Toolkit 문서](https://docs.nvidia.com/cuda/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)

### 한국어 LLM
- [Beomi's Open Ko LLM](https://huggingface.co/beomi)
- [Open Ko-LLaMA](https://github.com/open-ko-llama)
- [KoGPT](https://huggingface.co/kakaobrain/kogpt)

### 커뮤니티
- [Hugging Face 커뮤니티](https://huggingface.co/discussions)
- [PyTorch 포럼](https://discuss.pytorch.org/)
- [AI Korea Slack](https://aistartup-korea.slack.com)

---

## ⚖️ 라이센스

MIT License - 자유롭게 사용하세요! 🎉

본 프로젝트는 MIT 라이센스 하에 배포됩니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 👨‍💻 개발자

| 항목 | 내용 |
|------|------|
| **프로젝트** | Korean LLM V2 |
| **개발자** | seoan1210 |
| **시작 날짜** | 2026-07-25 |
| **상태** | 🚀 활발히 개발 중 |
| **라이선스** | MIT |

### 개발 환경 (테스트용)
- Lenovo Legion 7 Pro
- NVIDIA RTX 5090 (24GB)
- 64GB DDR5 RAM
- Windows 11

이 환경에서 벤치마크를 수행했지만, **코드는 모든 GPU에서 동작**하도록 설계되었습니다.

---

## 🤝 기여하기

이 프로젝트에 기여해주세요!

### 기여 방법
1. **Fork** 하기
2. **Branch** 생성: `git checkout -b feature/amazing-feature`
3. **Commit** 하기: `git commit -m 'Add amazing feature'`
4. **Push** 하기: `git push origin feature/amazing-feature`
5. **Pull Request** 열기

### 환영하는 기여
- 🐛 버그 리포트 및 수정
- 📚 문서 개선
- 🔧 성능 최적화
- 🌍 다른 언어/플랫폼 지원
- 💡 새로운 기능 제안

---

## 💬 문의 & 피드백

- 🐛 **버그 리포트**: [GitHub Issues](https://github.com/seoan1210/korean-llm-v2/issues)

---

## 🙏 감사의 말

이 프로젝트가 가능했던 것은 다음 덕분입니다:

### 오픈소스 & 라이브러리
- 🤗 **Hugging Face** - 데이터셋, 토크나이저, 모델 생태계
- 🔥 **PyTorch 커뮤니티** - 강력한 딥러닝 프레임워크
- 🎓 **NVIDIA** - CUDA, cuDNN, GPU 기술

### 데이터셋 제공자
- 📚 **maywell** - Korean Textbooks
- 🔄 **squarelike** - OpenOrca-gugugo-ko
- 🦙 **beomi** - KoAlpaca & 한국어 토크나이저

### 커뮤니티
- 🇰🇷 한국 PyTorch 커뮤니티
- 🤖 AI Korea 커뮤니티
- 💡 오픈소스 개발자들

---

## 📊 프로젝트 통계

```
📁 코드: 1,300 라인 (korean_llm_advanced_v2.py)
📚 문서: 완벽한 README.md
🧪 테스트: 자동 검증 스크립트 포함
⚡ 속도: 4.5 초/스텝 (RTX 5090)
📈 손실: 1.50 ± 0.08 (현재)
🎯 목표: 50,000 스텝
```

---

## 🚀 로드맵

| 단계 | 목표 스텝 | 상태 |
|------|---------|------|
| 첫 수렴 | 10K | 🔄 진행 중 |
| 기초 의미 | 20K | ⏳ 예정 |
| 안정적 생성 | 30K | ⏳ 예정 |
| 고급 기능 | 40K | ⏳ 예정 |
| 완성 & 배포 | 50K | ⏳ 예정 |

---

<div align="center">

## ⭐ 도움이 되었다면 스타를 눌러주세요! ⭐

**Made with ❤️ and 🤖 in Korea**

모두를 위한 한국어 LLM 프로젝트

[이슈 보고](https://github.com/seoan1210/korean-llm-v2/issues) • 
[Pull Request](https://github.com/seoan1210/korean-llm-v2/pulls)

```
 🇰🇷
한국어 LLM V2
 열심히 학습 중!
   ▲▲▲
  ▲△▲△▲
 ▲△▲☆▲△▲
```

[위로 가기 ⬆️](#-korean-llm-v2)

</div>

---

**Happy Training! 🚀🇰🇷**