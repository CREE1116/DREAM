# DreamV2: Recursive Reasoning Engine

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Apache--2.0-D22128?style=for-the-badge&logo=apache&logoColor=white" />
</p>

**DreamV2**는 **D**ynamic **RE**cursive **A**ttention **M**odel (**DREAM**) 아키텍처를 기반으로 하는 차세대 언어 모델 실험 프로젝트입니다. 모델이 하나의 토큰에 대해 마치 꿈을 꾸듯 내부적으로 재귀적인 사고를 반복하며 최적의 상태로 진화해가는 과정을 의미합니다. 

고정된 레이어 수를 통과하는 기존 방식에서 벗어나, 데이터의 복잡도에 따라 동적으로 연산량을 할당하는 지능형 엔진을 지향합니다.

---

## 💡 Why DREAM?

DreamV2는 단순한 언어 모델을 넘어 **'생각하는 효율'**에 집중합니다.

- **초경량 구조 (Efficiency)**: 단 3.15M의 블록 파라미터로 동작하면서도 반복 연산을 통해 깊은 추론 성능을 확보합니다.
- **적응적 연산 깊이 (Adaptive Depth)**: 쉬운 토큰은 빠르게 통과하고, 어려운 토큰은 더 깊게(Pondering) 처리하여 토큰별 최적의 연산량을 할당합니다.
- **Novel Step Encoding**: `Freeze ratio` 기반의 스텝 인코딩을 통해 모델이 현재 사고의 진척도를 스스로 인지하며 상태를 진화시킵니다.
- **Test-time Compute Scaling**: `tau` (임계값) 파라미터 하나만으로 추론 시 연산 깊이와 정확도 사이의 트레이드오프를 동적으로 조절할 수 있습니다.

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

## 📝 Examples (Hypothetical)

| 입력 | 출력 | 평균 추론 스텝 (ARS) |
|------|------|-----|
| 소크라테스는 말했다 | "너 자신을 알라"는 철학적 명언을 남겼습니다. | 16 |
| 철학이란 무엇인가 | 존재, 지식, 가치, 이성, 인식 등에 대한... | 77 |

---

## 📚 Related Work

DreamV2는 다음과 같은 연구들로부터 영감을 받았으며, 이를 독자적인 방식으로 개선하였습니다.

- **Universal Transformer**: 반복적인 레이어 통과를 제안했으나, DreamV2는 더 정교한 수렴 감지 알고리즘을 사용합니다.
- **PonderNet**: 동적 연산량 할당의 개념을 정립했으나, DreamV2는 **Freeze Mechanism**과 **Step Encoding**의 조합을 통해 수렴 안정성을 더 높였습니다.
- **Claude Mythos (OpenMythos)**: 유사한 철학을 공유하지만, DreamV2는 초경량 파라미터(3.15M) 환경에서의 극단적인 효율성에 집중합니다.

---

## 📖 Documentation

더 자세한 정보는 아래 문서들을 참고하세요:
- [Core Architecture](./docs/architecture.md): DREAM 엔진 및 신경망 구조 상세 설명
- [Usage Guide](./docs/usage.md): 설치 및 실행 방법 가이드

---

## 📜 License

DreamV2 프로젝트는 **Apache License 2.0** 하에 배포됩니다. 자유롭게 활용, 수정 및 배포가 가능합니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 확인하세요.

---
