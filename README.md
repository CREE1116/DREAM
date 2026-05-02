# DreamV2: Recursive Reasoning Engine

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white" />
</p>

**DreamV2**는 재귀적 추론(Dynamic REcursive Attention Model, DREAM)을 통해 사고하는 차세대 언어 모델 실험 프로젝트입니다. 고정된 레이어를 통과하는 기존 방식에서 벗어나, 데이터의 복잡도에 따라 동적으로 연산량을 할당하는 지능형 엔진을 지향합니다.

---

## ✨ Key Features

- **DREAM Engine**: 동일한 레이어를 동적으로 반복 통과하며 상태를 진화시키는 재귀적 추론 아키텍처.
- **Dynamic Pondering**: 코사인 유사도 기반 수렴 감지 시스템을 통해 문장별 최적 연산량 할당.
- **Modern Architecture**:
  - **RMSNorm & QK Norm**: 극대화된 학습 안정성과 연산 효율.
  - **RoPE (Rotary Positional Embedding)**: 정교한 상대적 위치 관계 학습.
  - **SwiGLU Activation**: FFN의 표현력 강화.
- **SFT Ready**: `<sys>`, `<usr>`, `<bot>` 통합 포맷을 지원하는 지시어 튜닝 파이프라인.
- **Efficient Data Pipeline**: 대규모 데이터셋을 위한 바이너리 패킹 및 메모리 맵핑 시스템.

---

## 📂 Project Structure

```text
DreamV2/
├── src/                # Core Architecture & Logic
│   ├── model.py        # DREAM Model implementation
│   └── data_loader.py  # Packed Dataset & DataLoader
├── scripts/
│   ├── train/          # Training Pipelines (PT, SFT, Tokenizer)
│   ├── inference/      # Chat & Generation scripts
│   └── explore/        # Model & Data analysis tools
├── docs/               # Detailed Documentation
├── checkpoints/        # Model Weights (Auto-generated)
└── data/               # Datasets & Cache (Auto-generated)
```

---

## 🚀 Quick Start

### 1. Requirements
본 프로젝트는 [uv](https://github.com/astral-sh/uv)를 사용하여 의존성을 관리합니다.

```bash
uv sync
```

### 2. Basic Workflow
상세한 설명은 [Usage Guide](./docs/usage.md)를 참고하세요.

```bash
# 1. 토크나이저 학습
uv run python scripts/train/tokenizer_train.py

# 2. 모델 사전 학습 (Pre-training)
uv run python scripts/train/train.py

# 3. 지시어 튜닝 (SFT)
uv run python scripts/train/sft.py

# 4. 채팅 테스트
uv run python scripts/inference/chat.py
```

---

## 📖 Documentation

더 자세한 정보는 아래 문서들을 참고하세요:
- [Core Architecture](./docs/architecture.md): DREAM 엔진 및 신경망 구조 상세 설명
- [Usage Guide](./docs/usage.md): 설치 및 실행 방법 가이드

---

## 📜 License

DreamV2 프로젝트는 **Apache License 2.0** 하에 배포됩니다. 자유롭게 활용, 수정 및 배포가 가능합니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---
