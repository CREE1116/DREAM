# DreamV2 Architecture

DreamV2는 재귀적 진화(Recursive Evolution)를 통해 사고하는 **DREAM (Dynamic REcursive Attention Model)** 엔진을 기반으로 합니다.

## 1. Core Concepts: DREAM Engine

DREAM의 핵심은 고정된 레이어 수를 통과하는 대신, 동일한 파라미터를 공유하는 단일 레이어를 **동적으로 여러 번 통과(Pondering)**하며 최적의 상태에 도달할 때까지 추론을 지속하는 것입니다.

### 1.1 Dynamic Pondering
- 모델은 매 스텝마다 이전 상태와 현재 상태의 코사인 유사도(Cosine Similarity)를 계산합니다.
- 유사도가 설정된 임계값(`tau`)을 넘어서면 해당 토큰은 '수렴'한 것으로 간주되어 더 이상의 연산을 멈춥니다(Freeze).
- 이를 통해 복잡한 문장에는 더 많은 연산량을, 쉬운 문장에는 적은 연산량을 동적으로 할당합니다.

### 1.2 Step Encoding
- 재귀적인 스텝이 진행됨에 따라 모델이 현재 어느 단계에 있는지 인지할 수 있도록 두 가지 정보를 결합한 Sinusoidal Embedding을 제공합니다:
  - `freeze_ratio`: 전체 토큰 중 수렴된 토큰의 비율
  - `step_progress`: 최대 허용 스텝 대비 현재 스텝의 위치

## 2. Model Architecture

DreamV2는 최신 대형 언어 모델(LLM)의 검증된 기술들을 채택하고 있습니다.

### 2.1 RMSNorm & QK Norm
- **RMSNorm**: LayerNorm 대비 연산 효율이 높고 수치적 안정성이 뛰어납니다.
- **QK Norm**: Query와 Key에 각각 RMSNorm을 적용하여 Attention Score의 폭주를 막고 학습 안정성을 극대화합니다.

### 2.2 Rotary Positional Embedding (RoPE)
- 절대적 위치 대신 상대적 위치 관계를 효과적으로 학습할 수 있는 RoPE를 적용하였습니다.

### 2.3 SwiGLU FFN
- 표준 ReLU/GELU 대신 SwiGLU 활성화 함수를 사용하여 Feed-Forward Network의 표현력을 높였습니다.

## 3. Data Pipeline
- **Packed Dataset**: 텍스트를 토큰화한 후 하나의 거대한 바이너리 파일(`.bin`)로 패킹합니다.
- **Memory Mapping**: `np.memmap`을 사용하여 대용량 데이터를 메모리에 올리지 않고 디스크에서 직접 읽어와 RAM 사용량을 최적화합니다.
