# Usage Guide

DreamV2 프로젝트의 설치부터 학습, 테스트까지의 전체 과정을 안내합니다.

## 1. Environment Setup

본 프로젝트는 `uv` 패키지 매니저를 사용하여 의존성을 관리합니다.

```bash
# 의존성 설치 및 가상환경 설정
uv sync
```

## 2. Tokenizer Training

모델 학습 전, 한국어 처리에 특화된 토크나이저를 먼저 생성해야 합니다.

```bash
# Byte-level BPE 토크나이저 학습
uv run python scripts/train/tokenizer_train.py

# (Optional) 토크나이저 분절 테스트
uv run python scripts/explore/test_tokenizer.py
```

## 3. Model Training (Pre-training)

방대한 양의 한국어 텍스트를 사용하여 베이스 모델을 학습시킵니다. 처음 실행 시 데이터를 `.bin` 파일로 패킹하는 과정이 포함됩니다.

```bash
# 사전 학습 시작
uv run python scripts/train/train.py
```
- 학습 현황은 `runs/` 폴더의 TensorBoard 로그를 통해 확인할 수 있습니다.
- 체크포인트는 `checkpoints/` 폴더에 스텝별로 저장됩니다.

## 4. Instruction Fine-tuning (SFT)

베이스 모델을 대화형 모델로 튜닝합니다. `heegyu/open-korean-instructions` 데이터셋의 `<sys>`, `<usr>`, `<bot>` 포맷을 사용합니다.

```bash
# SFT 학습
uv run python scripts/train/sft.py
```

## 5. Inference & Chat

학습된 모델과 직접 대화하며 성능을 확인합니다.

```bash
# 일반 텍스트 생성 테스트
uv run python scripts/inference/inference.py

# SFT 대화형 채팅 테스트
uv run python scripts/inference/chat.py
```

## 6. Monitoring Utilities

```bash
# 모델 파라미터 수 및 레이어 구조 확인
uv run python scripts/explore/check_param.py

# 패킹된 데이터셋의 총 토큰 수 확인
uv run python scripts/explore/check_data.py
```
