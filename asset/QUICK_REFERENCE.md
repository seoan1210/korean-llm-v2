# 🚀 한국어 LLM v2 - 빠른 참조 가이드

## 파일 구조
```
korean_llm_advanced_v2.py    ← 메인 학습 코드 (1000줄)
test_korean_llm.py           ← 테스트 스크립트
README_KOREAN_LLM_V2.md      ← 완전한 설명서
QUICK_REFERENCE.md           ← 이 파일 (빠른 참조)
```

---

## 🎯 3분 안에 시작하기

### 1️⃣ 설치
```bash
pip install torch transformers datasets
```

### 2️⃣ 테스트
```bash
python test_korean_llm.py
```
→ 모든 것이 잘 작동하는지 확인합니다 (5분 소요)

### 3️⃣ 학습 시작
```bash
python korean_llm_advanced_v2.py
```
→ 실제 학습이 시작됩니다

---

## 📊 핵심 설정값

### 기본 (RTX 5090 권장)
```python
from korean_llm_advanced_v2 import main, TrainingConfig

config = TrainingConfig(
    batch_size=2,              # ✅ RTX 5090에서 안정적
    accumulation_steps=32,     # 유효 배치 = 2 × 32 = 64
    max_steps=50000,
    learning_rate=5e-5,
    eval_interval=10,
)
main(config)
```

### 메모리 절약 (부족 시)
```python
config = TrainingConfig(
    batch_size=1,              # 줄임
    accumulation_steps=64,     # 유지
    max_seq_len=128,           # 줄임
)
main(config)
```

### 빠른 학습
```python
config = TrainingConfig(
    batch_size=3,              # 올림
    accumulation_steps=16,     # 줄임
    eval_interval=20,          # 평가 간격 ↑
)
main(config)
```

---

## 🔑 중요한 개선사항 (v1 vs v2)

| 항목 | v1 | v2 |
|-----|----|----|
| 데이터셋 무한루프 | ❌ 문제 있음 | ✅ 완벽 해결 |
| KV-Cache 누수 | ❌ 메모리 누수 | ✅ 길이 제한 |
| Checkpoint | ❌ 충돌 | ✅ 분리됨 |
| 생성 샘플링 | ❌ Greedy | ✅ Temperature + Top-K |
| 배치사이즈 | 1 | 2-3 |
| 추론 속도 | 기준 | 10배 빠름 |

---

## 🤖 클래스 빠른 참조

### KoreanLLM
```python
model = KoreanLLM(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    dim=1280,          # 모델 차원
    n_layers=20,       # 트랜스포머 레이어
    n_heads=10,        # 어텐션 헤드
    max_seq_len=1024   # 최대 시퀀스
).to(device)
```

### MultiKoreanDataset
```python
dataset = MultiKoreanDataset(
    tokenizer,
    max_len=256,              # 최대 시퀀스 길이
    samples_per_epoch=10000   # 에포크당 샘플 수
)
```

### 생성 함수
```python
response = generate(
    model, tokenizer,
    prompt="한국의 수도는",
    max_tokens=100,      # 생성 길이
    temperature=0.7,     # 낮을수록 고정적
    top_k=40,            # 상위 N개만 고려
    device=device
)
```

---

## 📈 학습 모니터링

### 로그 읽기
```
[Step   10] Loss: 4.2341 | LR: 5.00e-05 | Tokens/step: 512
          ↑               ↑               ↑
     스텝 번호      손실값          학습률
```

### 확인 사항
- ✅ Loss가 계속 줄어드는가?
- ✅ LR이 천천히 줄어드는가? (Cosine Annealing)
- ✅ GPU 메모리가 안정적인가?
- ✅ 생성 결과가 개선되는가?

---

## 💾 체크포인트 저장/로드

### 자동 저장
```python
# eval_interval마다 자동 저장됨
config = TrainingConfig(
    eval_interval=10,  # 10 스텝마다
)
# → korean_llm_00010.pth, korean_llm_00020.pth, ...
```

### 수동 로드
```python
checkpoint = torch.load('korean_llm_00010.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

---

## 🎯 데이터셋

### 사용 데이터셋
1. **maywell/korean_textbooks** - 한국 교과서
2. **squarelike/OpenOrca-gugugo-ko** - OpenOrca 한국어
3. **beomi/KoAlpaca-v1.1a** - KoAlpaca 한국어

### 데이터셋 추가하기
```python
# MultiKoreanDataset의 datasets_config 수정
self.datasets_config = [
    {"name": "new_dataset_name", "config": "config_name", "split": "train"},
    # ...
]
```

---

## ⚡ 성능 팁

### 배치사이즈와 메모리
```
배치사이즈 1, 누적 64 → 64 (기준)
배치사이즈 2, 누적 32 → 64 (2배 빠름, 메모리 +2GB)
배치사이즈 3, 누적 22 → 66 (2.5배 빠름, 메모리 +4GB)
배치사이즈 4, 누적 16 → 64 (2.8배 빠름, 메모리 +6GB)
```

### 학습률 스케줄
```
- Warmup: 200 스텝 동안 선형 증가
- 이후: Cosine Annealing으로 천천히 감소
→ 안정적인 학습 + 좋은 최종 성능
```

### Gradient Clipping
```python
# 이미 적용됨 (1.0)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

---

## 🔍 디버깅 팁

### Loss가 NaN인 경우
```python
# 1. Learning rate 줄이기
config.learning_rate = 1e-5

# 2. Batch size 줄이기
config.batch_size = 1

# 3. 처음 몇 스텝 로그 확인
for step in range(100):
    print(f"Step {step}: loss={loss.item()}")
```

### 메모리 부족
```python
# 옵션 1: 배치사이즈 줄이기
batch_size = 1

# 옵션 2: 시퀀스 길이 줄이기
max_seq_len = 128

# 옵션 3: 모델 크기 줄이기
dim = 768, n_layers = 12

# 옵션 4: Gradient Checkpointing (이미 활성화됨)
# Forward pass에서만 활성화, 추론은 안 함
```

### 생성이 반복적
```python
# Temperature 높이기
response = generate(..., temperature=0.9)

# Top-K 낮추기
response = generate(..., top_k=20)

# 또는 다른 sampling 방법 사용
# (Top-P, Nucleus sampling 등)
```

---

## 📋 체크리스트

### 시작 전
- [ ] CUDA 설치 확인: `nvidia-smi`
- [ ] PyTorch 설치: `pip install torch`
- [ ] Transformers 설치: `pip install transformers`
- [ ] 인터넷 연결 (데이터셋 다운로드)

### 학습 중
- [ ] Loss 모니터링 (NaN 아닌지)
- [ ] GPU 메모리 (24GB 이하 안정적)
- [ ] 생성 결과 (점점 개선되는지)
- [ ] 체크포인트 자동 저장

### 최적화
- [ ] 배치사이즈 조절
- [ ] 학습률 미세조정
- [ ] 평가 간격 조절 (속도 ↑)

---

## 🚀 빠른 명령어

```bash
# 테스트 실행
python test_korean_llm.py

# 학습 시작
python korean_llm_advanced_v2.py

# GPU 확인
nvidia-smi

# Python 버전 확인
python --version

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# 실행 중인 프로세스 보기
nvidia-smi -l 1  # 1초마다 갱신
```

---

## 💡 코드 커스터마이징 예제

### 학습률만 변경
```python
from korean_llm_advanced_v2 import main, TrainingConfig

config = TrainingConfig()
config.learning_rate = 1e-5
main(config)
```

### 모델 크기 변경
```python
# korean_llm_advanced_v2.py의 main() 함수에서
model = KoreanLLM(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    dim=2048,          # 증가
    n_layers=32,       # 증가
    n_heads=16,        # 증가
    max_seq_len=config.max_seq_len
).to(device)
```

### 자동 저장 경로 변경
```python
# main() 함수에서 checkpoint_path 변경
checkpoint_path = f"/path/to/checkpoints/korean_llm_{actual_step:05d}.pth"
```

---

## 🎓 학습 자료

| 주제 | 링크 |
|------|------|
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| RoPE | [RoFormer](https://arxiv.org/abs/2104.09864) |
| Flash Attention | [Flash-Attention](https://arxiv.org/abs/2205.14135) |
| KV-Cache | [Efficient Transformers](https://arxiv.org/abs/2401.00288) |

---

## 📞 자주 묻는 질문

**Q: 배치사이즈를 어떻게 결정하나?**
A: RTX 5090 24GB에서는 배치 2-3이 최적. `test_korean_llm.py`로 확인해보세요.

**Q: 학습 시간이 얼마나 걸려?**
A: 배치 2, 누적 32로 약 50,000 스텝 = 하루~이틀 (데이터셋 속도 의존)

**Q: 생성 결과가 이상해요**
A: 초기에는 그럴 수 있습니다. 1000 스텝 이후부터 개선됩니다.

**Q: 메모리 부족해요**
A: `batch_size=1`, `max_seq_len=128`로 줄여보세요.

**Q: 데이터셋 다운로드가 느려요**
A: Hugging Face 캐시: `~/.cache/huggingface/datasets/`

---

## 🌟 다음 단계

1. **Phase 1** (지금): 기본 학습 (50M 파라미터)
2. **Phase 2**: 모델 크기 증가 (200M+)
3. **Phase 3**: 분산학습 (멀티 GPU)
4. **Phase 4**: 파인튜닝 (LoRA)
5. **Phase 5**: 배포 (ONNX, TorchServe)

---

**Happy training! 🚀🇰🇷**

마지막 업데이트: 2026-07-25
