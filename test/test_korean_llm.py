#!/usr/bin/env python3
"""
빠른 테스트 스크립트
- 모델 생성 확인
- 데이터셋 로드 확인
- 포워드 패스 테스트
- 생성 테스트
- 메모리 사용량 확인

실행: python test_korean_llm.py
시간: ~2-5분
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import logging
from korean_llm_advanced_v2 import (
    KoreanLLM,
    MultiKoreanDataset,
    generate,
    TrainingConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_header(text):
    """헤더 출력"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_success(text):
    """성공 메시지"""
    print(f"✅ {text}")

def print_error(text):
    """에러 메시지"""
    print(f"❌ {text}")

def test_device():
    """GPU 확인"""
    print_header("1️⃣ GPU 확인")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_success(f"CUDA 사용 가능!")
        print(f"  - 디바이스: {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"  - PyTorch CUDA: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print_error("CUDA를 사용할 수 없습니다. CPU로 진행합니다.")
    
    return device

def test_tokenizer():
    """토크나이저 테스트"""
    print_header("2️⃣ 토크나이저 로드")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "beomi/Llama-3-Open-Ko-8B",
            clean_up_tokenization_spaces=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        print_success(f"토크나이저 로드 완료")
        print(f"  - 보캐브 크기: {len(tokenizer)}")
        print(f"  - EOS 토큰 ID: {tokenizer.eos_token_id}")
        print(f"  - PAD 토큰 ID: {tokenizer.pad_token_id}")
        
        # 샘플 인코딩
        text = "안녕하세요! 한국어 LLM 테스트입니다."
        tokens = tokenizer.encode(text, return_tensors="pt")
        print(f"  - 샘플 텍스트: '{text}'")
        print(f"  - 토큰 수: {tokens.shape[1]}")
        
        return tokenizer
    except Exception as e:
        print_error(f"토크나이저 로드 실패: {e}")
        raise

def test_model_creation(device, tokenizer):
    """모델 생성 테스트"""
    print_header("3️⃣ 모델 생성")
    
    try:
        model = KoreanLLM(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            dim=512,              # 테스트용 작은 크기
            n_layers=4,           # 테스트용 작은 크기
            n_heads=8,
            max_seq_len=256
        ).to(device)
        
        # 모델 크기 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_success("모델 생성 완료")
        print(f"  - 총 파라미터: {total_params / 1e6:.1f}M")
        print(f"  - 학습 가능: {trainable_params / 1e6:.1f}M")
        print(f"  - 디바이스: {device}")
        
        return model
    except Exception as e:
        print_error(f"모델 생성 실패: {e}")
        raise

def test_forward_pass(device, model, tokenizer):
    """포워드 패스 테스트"""
    print_header("4️⃣ 포워드 패스 테스트")
    
    try:
        # 더미 토큰 생성
        batch_size = 2
        seq_len = 64
        tokens = torch.randint(0, len(tokenizer), (batch_size, seq_len)).to(device)
        labels = torch.randint(0, len(tokenizer), (batch_size, seq_len)).to(device)
        
        print(f"  - 입력 크기: {tokens.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            logits, loss, kv_caches = model(tokens, labels=labels)
        
        print_success("포워드 패스 완료")
        print(f"  - 출력 크기: {logits.shape}")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - KV-Cache 길이: {len(kv_caches)}")
        
        # 메모리 확인
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"  - GPU 메모리 할당: {allocated:.2f}GB")
            print(f"  - GPU 메모리 예약: {reserved:.2f}GB")
        
        return logits, loss
    except Exception as e:
        print_error(f"포워드 패스 실패: {e}")
        raise

def test_backward_pass(device, model, tokenizer):
    """역전파 테스트"""
    print_header("5️⃣ 역전파 테스트")
    
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 더미 데이터
        tokens = torch.randint(0, len(tokenizer), (2, 64)).to(device)
        labels = tokens.clone()
        
        # Forward
        logits, loss, _ = model(tokens, labels=labels)
        
        # Backward
        loss.backward()
        
        print_success("역전파 완료")
        print(f"  - Loss: {loss.item():.4f}")
        
        # 그래디언트 확인
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if name.endswith('weight') and grad_norm > 0:
                    print(f"  - {name}: grad_norm={grad_norm:.4f}")
                    break
        
        if has_grad:
            print_success("그래디언트 계산 완료")
        else:
            print_error("그래디언트가 계산되지 않았습니다")
        
        # 옵티마이저 스텝
        optimizer.step()
        optimizer.zero_grad()
        
        return True
    except Exception as e:
        print_error(f"역전파 실패: {e}")
        raise

def test_generation(device, model, tokenizer):
    """생성 테스트"""
    print_header("6️⃣ 생성 테스트")
    
    try:
        model.eval()
        
        prompts = [
            "안녕?",
            "한국의 수도는",
            "인공지능이란"
        ]
        
        for prompt in prompts:
            print(f"\n  📝 프롬프트: '{prompt}'")
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
                top_k=20,
                device=device
            )
            print(f"  🤖 응답: '{response}'")
        
        print_success("생성 완료")
        return True
    except Exception as e:
        print_error(f"생성 실패: {e}")
        return False

def test_dataset():
    """데이터셋 로드 테스트"""
    print_header("7️⃣ 데이터셋 테스트")
    
    try:
        # 토크나이저 먼저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            "beomi/Llama-3-Open-Ko-8B",
            clean_up_tokenization_spaces=False
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # 데이터셋 생성
        dataset = MultiKoreanDataset(
            tokenizer,
            max_len=128,
            samples_per_epoch=10
        )
        
        print_success("데이터셋 생성 완료")
        
        # 몇 개 샘플 로드
        print("\n  📚 샘플 데이터:")
        iterator = iter(dataset)
        for i in range(3):
            sample = next(iterator)
            print(f"    - 샘플 {i+1}: 토큰 형태 {sample.shape}")
        
        return True
    except Exception as e:
        print_error(f"데이터셋 로드 실패: {e}")
        logger.error(f"자세한 오류: {e}", exc_info=True)
        return False

def test_full_training_loop(device, tokenizer):
    """전체 학습 루프 테스트 (1 스텝)"""
    print_header("8️⃣ 전체 학습 루프 테스트")
    
    try:
        # 작은 모델로 테스트
        model = KoreanLLM(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            dim=256,
            n_layers=2,
            n_heads=4,
            max_seq_len=128
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # 더미 배치
        batch = torch.randint(0, len(tokenizer), (2, 128)).to(device)
        
        # Forward
        model.train()
        logits, loss, _ = model(batch, labels=batch)
        
        # Backward
        loss.backward()
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        print_success("학습 루프 완료 (1 스텝)")
        print(f"  - 초기 Loss: {loss.item():.4f}")
        
        # GPU 메모리 정보
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(device) / 1e9
            print(f"  - 최종 GPU 메모리: {allocated:.2f}GB")
        
        return True
    except Exception as e:
        print_error(f"학습 루프 실패: {e}")
        logger.error(f"자세한 오류: {e}", exc_info=True)
        return False

def main():
    """메인 테스트 함수"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + " 🚀 한국어 LLM v2 - 테스트 스위트".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        # 1. GPU 확인
        device = test_device()
        
        # 2. 토크나이저 테스트
        tokenizer = test_tokenizer()
        
        # 3. 모델 생성
        model = test_model_creation(device, tokenizer)
        
        # 4. 포워드 패스
        test_forward_pass(device, model, tokenizer)
        
        # 5. 역전파
        test_backward_pass(device, model, tokenizer)
        
        # 6. 생성
        test_generation(device, model, tokenizer)
        
        # 7. 데이터셋 (선택)
        print("\n" + "="*60)
        print("8️⃣ 데이터셋 테스트는 인터넷 연결 필요합니다")
        print("건너뛰려면 y를 입력하세요.")
        user_input = input("데이터셋 테스트를 진행할까요? (y/n): ").strip().lower()
        if user_input != 'n':
            test_dataset()
        
        # 8. 전체 학습 루프
        test_full_training_loop(device, tokenizer)
        
        # 결과 요약
        print_header("✨ 모든 테스트 완료!")
        print("\n✅ 축하합니다! 모든 테스트를 통과했습니다!")
        print("\n📝 다음 단계:")
        print("  1. python korean_llm_advanced_v2.py 를 실행하여 실제 학습을 시작하세요")
        print("  2. README_KOREAN_LLM_V2.md 를 읽고 설정을 커스터마이즈하세요")
        print("  3. 체크포인트를 정기적으로 저장하고 모니터링하세요")
        print("\n🚀 Happy training! 🇰🇷\n")
        
    except Exception as e:
        print_header("❌ 테스트 실패!")
        print_error(f"심각한 오류가 발생했습니다: {e}")
        logger.error(f"전체 오류 정보:", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
