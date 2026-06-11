# World-Model-Zoo 🌍

Deep model-based RL의 주요 world model 계보를 PyTorch로 자체 구현하고,
축소 벤치마크에서 논문 절대 점수를 재현하는 프로젝트.

공식 구현을 참조하지 않고 논문+부록만으로 구현한 뒤, 후반부에 레퍼런스와 대조하여
"논문에 없는 트릭"을 발굴하는 것을 목표로 한다.

---

## 구현 대상

| # | 알고리즘 | 논문 | 핵심 기여 | 상태 |
|---|---|---|---|---|
| 1 | **RSSM** (PlaNet) | Hafner et al., 2019 | Recurrent State-Space Model 기반 latent dynamics | 🔲 |
| 2 | **Dreamer v1** | Hafner et al., 2020 | Latent imagination으로 actor-critic 학습 | 🔲 |
| 3 | **Dreamer v2** | Hafner et al., 2021 | Categorical latent + KL balancing, discrete action | 🔲 |
| 4 | **Dreamer v3** | Hafner et al., 2023/2025 | Symlog / two-hot / 단일 하이퍼파라미터로 범용화 | 🔲 |
| 5 | **TD-MPC2** | Hansen et al., 2024 | Reconstruction-free, latent consistency + MPPI | 🔲 |
| 6 | **IRIS** | Micheli et al., 2023 | VQ-VAE 토크나이저 + GPT autoregressive dynamics | 🔲 |
| 7 | **STORM** | Zheng et al., 2024 | Categorical VAE + Transformer dynamics | 🔲 |
| 8 | **DIAMOND** | Alonso et al., 2024 | Diffusion 기반 pixel-space world model | 🔲 stretch |

---

## 벤치마크 & 재현 기준

논문 보고 절대 점수 **±15% 이내, 3 seeds 평균** 도달을 재현 성공으로 정의한다.
타깃 점수 원본: [`docs/TARGETS.md`](docs/TARGETS.md)

| 도메인 | 환경 | 예산 | 대상 알고리즘 |
|---|---|---|---|
| Atari 100k | Pong, Breakout, Boxing | 100k agent steps | DreamerV2/V3, IRIS, STORM, DIAMOND |
| DMC (proprio) | walker-walk, cheetah-run | 500k env steps | DreamerV1, TD-MPC2 |
| DMC (vision) | walker-walk | 500k env steps | DreamerV3, TD-MPC2 |

---

## 결과

> 학습 완료 후 채워짐.

| 알고리즘 | Pong | Breakout | Boxing | walker-walk | cheetah-run |
|---|---|---|---|---|---|
| DreamerV1 | — | — | — | 🔲 | 🔲 |
| DreamerV2 | 🔲 | 🔲 | 🔲 | — | — |
| DreamerV3 | 🔲 | 🔲 | 🔲 | 🔲 | — |
| TD-MPC2 | — | — | — | 🔲 | 🔲 |
| IRIS | 🔲 | 🔲 | 🔲 | — | — |
| STORM | 🔲 | 🔲 | 🔲 | — | — |
| DIAMOND | 🔲 | — | — | — | — |

---

## 구조

```
world-model-zoo/
├── wm/
│   ├── common/          # 공유 인프라 (환경 래퍼, 버퍼, 로거, 평가, 시각화)
│   ├── algos/           # 알고리즘별 구현 (rssm, dreamer_v1, ..., diamond)
│   └── configs/         # 알고리즘×환경 yaml (논문 부록 하이퍼파라미터 기준)
├── scripts/             # train.py, eval.py, collect_random.py 등
├── docs/
│   ├── ROADMAP.md       # task 단위 진행 로드맵
│   ├── TARGETS.md       # 알고리즘×환경 타깃 점수 테이블
│   └── notes/           # 알고리즘별 구현 노트 (논문에 없는 트릭 포함)
└── results/             # 학습 곡선, 점수 테이블, open-loop GIF
```

---

## 설치

```bash
# uv 설치 (없는 경우)
curl -Lsf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync

# Intel Arc (XPU) 환경인 경우
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

디바이스는 `cuda → xpu → mps → cpu` 순으로 자동 감지된다.

---

## 실행

```bash
# 랜덤 정책으로 데이터 수집 (Task 2 이후)
uv run python scripts/collect_random.py --env walker-walk --steps 10000

# 학습
uv run python scripts/train.py --algo dreamer_v3 --env walker-walk --seed 0

# 체크포인트에서 재개 (Colab 세션 끊김 후)
uv run python scripts/train.py --algo dreamer_v3 --env walker-walk --seed 0 --resume

# 평가
uv run python scripts/eval.py --algo dreamer_v3 --env walker-walk --ckpt results/dreamer_v3/walker-walk/seed0/

# 스모크 테스트
uv run python scripts/smoke.py
uv run pytest
```

---

## Colab에서 실행

```python
!git clone https://github.com/<username>/world-model-zoo
%cd world-model-zoo
!pip install uv
!uv sync
!uv run python scripts/train.py --algo dreamer_v3 --env pong --seed 0
```

체크포인트는 Google Drive에 마운트해서 저장 권장 (`--ckpt-dir /content/drive/MyDrive/wm-results/`).

---

## 진행 방식

- **레퍼런스 없이 구현**: Task 12 이전에는 공식 구현 코드를 열람하지 않는다.
- **검증 순서**: world model open-loop 예측 → reward 예측 → RL 점수 순으로 디버깅.
- **알고리즘별 노트**: `docs/notes/<algo>.md`에 구현 중 발견한 논문-코드 간 디테일 차이를 기록.
- 상세 작업 계획: [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## 참고 논문

- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122)
- [PlaNet (Hafner et al., 2019)](https://arxiv.org/abs/1811.04551)
- [Dreamer v1 (Hafner et al., 2020)](https://arxiv.org/abs/1912.01603)
- [Dreamer v2 (Hafner et al., 2021)](https://arxiv.org/abs/2010.02193)
- [Dreamer v3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104)
- [TD-MPC2 (Hansen et al., 2024)](https://arxiv.org/abs/2310.16828)
- [IRIS (Micheli et al., 2023)](https://arxiv.org/abs/2209.14430)
- [STORM (Zheng et al., 2024)](https://arxiv.org/abs/2310.09615)
- [DIAMOND (Alonso et al., 2024)](https://arxiv.org/abs/2405.12399)