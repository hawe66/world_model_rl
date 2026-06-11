# World Model 계보 재현 프로젝트 — Task Roadmap

> Claude Code 소비용 마스터 로드맵. 각 task는 독립 세션에서 이 문서 + 해당 task 섹션을 컨텍스트로 주고 진행한다.
> 주차 단위가 아닌 task 단위로 유동적으로 진행하며, 선행 task의 DoD 충족이 다음 task 착수 조건이다.

---

## 프로젝트 원칙 (모든 task에 적용)

- **구현 언어**: PyTorch 자체 구현. 공식/서드파티 레퍼런스 코드는 Task 12 이전에는 열어보지 않는다 (논문 + 부록만 허용).
- **재현 기준**: 축소 벤치마크(Atari 100k 서브셋, DMC 서브셋)에서 **논문 보고 절대 점수의 ±15% 이내** 도달을 1차 목표, 알고리즘 간 상대 순위 재현을 2차 목표로 한다.
- **시드**: 알고리즘당 최소 3 seeds. 점수는 mean ± std로 보고.
- **코드 구조**: 알고리즘별 디렉토리는 `wm/algos/<name>/`에 격리하되, 환경/버퍼/로깅/평가는 `wm/common/`을 공유한다. 공유 모듈을 알고리즘 사정으로 수정할 때는 반드시 하위 호환 유지.
- **검증 우선순위**: world model 자체의 open-loop 예측 품질 → reward/continue 예측 정확도 → imagination 기반 policy 성능 순서로 디버깅한다. RL 점수가 안 나오면 항상 모델 예측 품질부터 본다.

### 디렉토리 구조 (Task 0에서 생성)

```
world-model-zoo/
├── pyproject.toml          # uv 기반
├── wm/
│   ├── common/
│   │   ├── envs/           # gymnasium 래퍼 (atari, dmc)
│   │   ├── buffer.py       # sequence replay buffer
│   │   ├── logger.py       # tensorboard + 메트릭 jsonl
│   │   ├── eval.py         # 고정 평가 프로토콜
│   │   └── viz.py          # open-loop rollout 시각화 (GIF/grid)
│   ├── algos/
│   │   ├── rssm/           # Task 4 (공유 모듈로 승격 가능)
│   │   ├── dreamer_v1/
│   │   ├── dreamer_v2/
│   │   ├── dreamer_v3/
│   │   ├── tdmpc2/
│   │   ├── iris/
│   │   ├── storm/
│   │   └── diamond/        # stretch
│   └── configs/            # 알고리즘×환경별 yaml
├── docs/
│   ├── SURVEY.md           # Task 1 산출물
│   ├── TARGETS.md          # Task 1 산출물 (절대 점수 타깃 테이블)
│   └── notes/<algo>.md     # 알고리즘별 구현 노트
├── scripts/                # train/eval 엔트리포인트
└── results/                # 점수 테이블, 곡선, GIF
```

### 고정 벤치마크

| 도메인 | 환경 | 예산 | 사용 알고리즘 |
|---|---|---|---|
| Atari 100k | Pong, Breakout, Boxing | 100k env steps (400k frames) | DreamerV2/V3, IRIS, STORM, DIAMOND |
| DMC (proprio) | walker-walk, cheetah-run | 500k env steps | DreamerV1, TD-MPC2 |
| DMC (vision) | walker-walk | 500k env steps | DreamerV3, TD-MPC2 |

평가: 학습 중 주기적으로 10 episodes 평균, 최종 점수는 마지막 체크포인트 기준 30 episodes 평균.

---

## Task 0 — 리포 부트스트랩

**목표**: 빈 리포에서 학습 1 step이 도는 골격까지.

**작업 항목**
- uv 기반 `pyproject.toml`: torch, gymnasium[atari,accept-rom-license], dm_control, tensorboard, einops, hydra-core(또는 단순 yaml+dataclass), pytest, ruff
- 위 디렉토리 구조 생성, `wm/` 패키지화
- GPU/MPS/XPU 감지 유틸(`wm/common/device.py`), 전역 seed 고정 유틸
- pre-commit (ruff format/lint), `pytest` 빈 테스트 1개 통과
- `scripts/smoke.py`: 더미 환경에서 랜덤 텐서 1 step forward/backward
- **체크포인트/재개 유틸** (`wm/common/checkpoint.py`): `save_checkpoint` / `load_checkpoint` / `--resume` 플래그 — Task 0에서 설계하고 모든 train script가 이를 사용 (DreamerV3 run이 6시간+이므로 Colab 12시간 세션 내 완주가 불가능 → 처음부터 설계 필수)

**DoD**:
- `uv run pytest` 통과
- `uv run python scripts/smoke.py`가 감지된 디바이스(CUDA/XPU/MPS/CPU)에서 동작
- `save_checkpoint` → `load_checkpoint` 왕복 pytest 통과
- `docs/GPU_CLOUD_SETUP.md` 존재 (Colab/Kaggle/vast.ai 워크플로우 및 --resume 설계 문서화)

**클라우드 학습 참고**: `docs/GPU_CLOUD_SETUP.md` 참조.
무료 티어(Colab/Kaggle)로는 3 seeds × 전체 run을 완주할 수 없다. 단일 seed 검증은 Kaggle(주 30h GPU), 전체 재현 run은 Colab Pro 또는 vast.ai/RunPod($0.2~0.4/h)를 권장.
노트북 셀에 코드를 짜지 말 것 — Colab/Kaggle은 `!git clone` + `!python scripts/train.py ... --resume` 실행만 한다.

---

## Task 1 — 계보 문서화 + 타깃 점수 확정

**목표**: 구현 대상과 "재현 성공"의 수치적 정의를 못 박는다.

**작업 항목**
- `docs/SURVEY.md`: World Models(2018) → PlaNet → Dreamer v1/v2/v3 → TD-MPC/TD-MPC2 → IRIS → STORM → DIAMOND 계보를 핵심 기여 1줄 + 아키텍처 다이어그램(mermaid) + 선행 대비 diff 관점으로 정리. tsinghua-fib-lab/World-Model survey 레포를 보조 자료로 활용.
- `docs/TARGETS.md`: **각 논문 원문 결과 테이블에서** 고정 벤치마크 해당 환경의 절대 점수를 추출해 테이블화. 점수가 논문에 없는 조합(예: TD-MPC2의 walker-walk 500k 시점)은 논문 그래프에서 읽은 근사치로 적고 출처 표기. 이 테이블이 전 프로젝트의 ±15% 기준선이 된다.
- 각 알고리즘의 논문 하이퍼파라미터(부록 테이블)를 `wm/configs/`에 yaml로 미리 옮겨 적는다.

**DoD**: TARGETS.md에 알고리즘×환경 전 조합의 타깃 점수와 출처(논문 테이블 번호)가 기재됨.

**주의**: 점수는 반드시 논문 원문에서 추출한다. 2차 자료(블로그, 타 논문 인용 테이블)는 검증용으로만.

---

## Task 2 — 환경 래퍼

**목표**: 모든 알고리즘이 동일한 관측/행동 인터페이스를 쓰게 한다.

**작업 항목**
- Atari: gymnasium ALE 기반. 64×64 grayscale(알고리즘별 설정으로 RGB 선택 가능), action repeat 4, max-pool over 2 frames, life-loss 처리 옵션, sticky action **off** (Atari 100k 표준), 100k env step 카운팅 명확화 (frame 기준 vs step 기준 혼동 금지 — 100k agent steps = 400k frames)
- DMC: proprio / vision(64×64 RGB, action repeat 2) 양쪽 지원
- 공통 인터페이스: `obs(dict: image|state), action, reward, is_first, is_last, is_terminal` — Dreamer 계열의 episode boundary 처리(continue flag)를 처음부터 반영
- 결정성 테스트: 같은 seed로 두 번 굴려 trajectory 일치 확인하는 pytest

**DoD**: `scripts/collect_random.py`로 Pong과 walker-walk에서 랜덤 정책 에피소드를 수집해 npz로 저장 가능. 결정성 테스트 통과.

---

## Task 3 — 버퍼 / 로깅 / 평가 / 시각화

**목표**: 알고리즘 코드가 학습 로직에만 집중할 수 있는 공통 인프라.

**작업 항목**
- `buffer.py`: 시퀀스 단위 샘플링 replay buffer. (uniform episode → uniform start index, 길이 L 청크, is_first 마스킹). 온라인 학습용 ring buffer + 오프라인 npz 로딩 양쪽 지원. 메모리 한계 대비 uint8 이미지 저장.
- `logger.py`: tensorboard + `metrics.jsonl` 이중 기록. step 축은 env step으로 통일.
- `eval.py`: 고정 평가 프로토콜 함수화 (eval env 분리, deterministic/stochastic policy 모드 선택).
- `viz.py`: ① open-loop rollout grid — context k step 주고 이후 n step을 모델 예측 vs 실제 프레임 나란히, ② imagination GIF. 모든 world model task의 1차 검증 도구.
- 버퍼 샘플링 분포/마스킹에 대한 pytest.

**DoD**: 랜덤 수집 데이터를 버퍼에 넣고 (B, L) 시퀀스 배치를 뽑아 시각화 grid를 저장하는 데모 스크립트 동작.

---

## Task 4 — RSSM 단독 구현 + open-loop 검증

**목표**: Dreamer 계보의 코어를 RL 없이 순수 시퀀스 모델링 문제로 먼저 검증.

**작업 항목**
- PlaNet 논문 기준 RSSM: deterministic path (GRU) + stochastic latent (Gaussian), CNN encoder/decoder, reward head
- 손실: reconstruction NLL + KL(posterior ‖ prior), free nats
- 학습 데이터: Task 2에서 수집한 랜덤 정책 rollout (walker-walk vision)
- 검증: ① 5-step context → 45-step open-loop 예측 grid가 육안으로 동역학을 따라가는가, ② reward 예측 MSE가 학습 곡선에서 단조 감소하는가, ③ KL이 0으로 붕괴(posterior collapse)하지 않는가
- `docs/notes/rssm.md`에 학습 안정성 관련 발견 기록 (KL balancing 부재 시 증상 등 — v2 task의 복선)

**DoD**: open-loop 예측 GIF에서 walker의 운동이 20 step 이상 그럴듯하게 유지됨. 이 모듈이 이후 dreamer_v1~v3의 공유 베이스가 된다.

---

## Task 5 — Dreamer v1 (DMC)

**목표**: imagination 기반 actor-critic을 RSSM 위에 얹어 첫 RL 점수 달성.

**작업 항목**
- actor: tanh-Gaussian policy, critic: λ-return 타깃 value
- imagination horizon H=15, dynamics backprop을 통한 actor 학습 (straight-through 아님 — v1은 reparameterized 연속 latent라 직접 미분)
- 온라인 학습 루프: env step ↔ train step 비율(train ratio) config화
- 환경: walker-walk, cheetah-run (vision)
- 타깃: TARGETS.md의 DreamerV1 점수 ±15%

**DoD**: walker-walk 3 seeds 평균이 타깃 범위 진입. 학습 곡선 + imagination GIF를 `results/dreamer_v1/`에 저장.

**예상 함정**: value 타깃 detach 누락, imagination 시작 state에 terminal state 포함, action repeat 불일치.

---

## Task 6 — Dreamer v2 (Atari)

**목표**: v1 → v2 diff 구현으로 discrete control 정복.

**작업 항목 (v1 대비 diff)**
- stochastic latent: Gaussian → 32×32 categorical + straight-through gradient
- KL balancing (prior/posterior 가중 분리)
- actor: discrete action + REINFORCE(+ entropy), dynamics backprop 혼합 비율
- reward/discount head로 episode 종료 예측
- 환경: Pong, Breakout, Boxing @ 100k
- 타깃: TARGETS.md의 DreamerV2 Atari 100k 점수 ±15% (단, v2는 원논문이 200M 프레임 기준이므로 100k 점수는 후속 논문들의 baseline 테이블 수치를 타깃으로 — Task 1에서 출처 확정)

**DoD**: Pong에서 3 seeds 평균 타깃 진입, Breakout/Boxing 결과 기록.

---

## Task 7 — Dreamer v3 (Atari + DMC, 절대 점수 재현의 본진)

**목표**: 단일 하이퍼파라미터 셋으로 두 도메인 모두 타깃 점수 도달.

**작업 항목 (v2 대비 diff)**
- symlog 예측, two-hot reward/value 인코딩
- free bits (KL 하한 1 nat), percentile 기반 return normalization
- critic EMA regularizer, replay 비율 등 v3 논문(Nature 2025 버전) 세부 반영
- 동일 config로 Atari 3종 + walker-walk(vision) 학습
- 타깃: DreamerV3 논문 보고 점수 ±15%

**DoD**: 4개 환경 모두 3 seeds 평균 타깃 진입. v1/v2/v3 곡선 비교 플롯 생성 — 계보 내 개선이 우리 구현에서도 재현되는지 확인.

---

## Task 8 — TD-MPC2 (DMC)

**목표**: reconstruction-free + planning 계열 대표 구현.

**작업 항목**
- latent consistency loss 기반 world model (decoder 없음), SimNorm latent
- MPPI planning + learned policy prior, ensemble Q
- 환경: walker-walk, cheetah-run (proprio 먼저, vision은 stretch)
- 타깃: TD-MPC2 논문 점수 ±15%

**DoD**: proprio 2개 task 타깃 진입. `docs/notes/tdmpc2.md`에 Dreamer 계열과의 구조적 차이(decoder 유무가 표현 학습에 미치는 영향) 관찰 기록.

---

## Task 9 — IRIS (Atari)

**목표**: Transformer world model 분기 1 — 토큰화 접근.

**작업 항목**
- VQ-VAE 이미지 토크나이저 (4×4×K tokens), 토크나이저 단독 사전 검증 (reconstruction 품질)
- GPT 스타일 autoregressive dynamics (token-level), reward/termination head
- imagination에서 policy 학습 (actor-critic)
- 환경: Pong, Breakout, Boxing @ 100k
- 타깃: IRIS 논문 Atari 100k 점수 ±15%

**DoD**: 3 게임 결과 기록, 최소 2개 타깃 진입. 토큰화 vs RSSM의 open-loop 예측 품질 비교 grid 생성.

---

## Task 10 — STORM (Atari)

**목표**: Transformer 분기 2 — categorical latent + Transformer 결합.

**작업 항목**
- DreamerV2식 categorical VAE + Transformer dynamics (IRIS와의 차이: per-frame latent vs per-patch token)
- 환경/타깃: Atari 3종 @ 100k, STORM 논문 점수 ±15%

**DoD**: 3 게임 결과 기록. DreamerV3 / IRIS / STORM 3자 비교 테이블 — 논문상 상대 순위가 재현되는지 확인.

---

## Task 11 (stretch) — DIAMOND (Atari)

**목표**: diffusion world model. 연산 비용이 가장 크므로 stretch로 분류.

**작업 항목**
- EDM 스타일 diffusion으로 다음 프레임 예측 (pixel space), 소수 denoising step
- 환경: Pong 단일로 축소 가능
- 타깃 미달 시에도 open-loop 시각 품질 비교(RSSM/IRIS 대비)만으로 산출물 인정

**DoD**: Pong 학습 1 seed 완주 + 시각 품질 비교 자료.

---

## Task 12 — 레퍼런스 대조 + 점수 갭 분석

**목표**: 이 시점에서 처음으로 공식/서드파티 구현을 연다.

**작업 항목**
- 대조 대상: dreamerv3 공식(JAX), dreamerv3-torch, TD-MPC2 공식, IRIS 공식, STORM 공식, SheepRL
- 타깃 미달 알고리즘에 대해: 하이퍼파라미터 diff → 전처리 diff → 손실 구현 diff 순으로 갭 원인 추적, 수정 후 재학습
- 우리 구현과 공식 구현의 미묘한 차이(논문에 없는 트릭)를 `docs/notes/`에 알고리즘별로 기록 — 이 문서가 포트폴리오에서 가장 차별화되는 자산

**DoD**: 전 알고리즘 최종 점수 확정, 갭 원인 분석 문서 완성.

---

## Task 13 — 최종 리포트

**작업 항목**
- `results/FINAL.md`: 알고리즘×환경 점수 테이블 (우리 구현 vs 논문, ±% 표기), 학습 곡선 통합 플롯, open-loop 예측 비교 grid/GIF
- 계보 관점 회고: 각 세대 전환에서 어떤 diff가 점수를 만들었는지 ablation적 서술
- 블로그/포트폴리오용 정리 (선택)

**DoD**: 리포 README에서 전체 결과에 도달 가능한 상태.

---

## 컴퓨팅 예산 메모

절대 점수 재현 + 3 seeds 조건이므로 가장 비용이 큰 축은 학습 횟수다. 대략적 단일 run 예상 (RTX 4090 기준 자릿수 감각):
- DreamerV3 Atari 100k: 6~12시간/run → 3게임×3seeds ≈ 3~5일
- IRIS: 토크나이저+GPT라 Dreamer보다 무거움, 12~24시간/run
- TD-MPC2 proprio: 가장 가벼움, 2~4시간/run
- DIAMOND: 가장 무거움 → stretch 사유

따라서 task 진행 중에도 "1 seed 먼저 완주 → 타깃 근접 확인 → 나머지 seeds 병렬"의 순서를 지킬 것.