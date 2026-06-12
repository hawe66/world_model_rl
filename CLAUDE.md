# WORLD_MODEL_RL — CLAUDE.md

Deep model-based RL world model 계보(PlaNet→Dreamer v1/v2/v3, TD-MPC2, IRIS, STORM, DIAMOND)를 PyTorch로 자체 구현하고 축소 벤치마크에서 논문 절대 점수를 재현하는 프로젝트. (패키지명: `world-model-zoo`)

## 마스터 문서

- 전체 로드맵: `docs/WORLD_MODEL_ROADMAP.md` — 작업은 항상 task 단위로 진행. 착수 전 해당 task 섹션을 읽을 것.
- 클라우드 학습 워크플로우: `docs/GPU_CLOUD_SETUP.md` — Colab/Kaggle/vast.ai 운용 및 `--resume` 설계.
- task별 상세 구현 플랜: `docs/superpowers/plans/` — 존재하면 로드맵보다 우선.
- 계보 서베이: `docs/SURVEY.md` — 알고리즘별 핵심 기여·아키텍처·선행 대비 diff.
- 타깃 점수: `docs/TARGETS.md` — 재현 성공 기준은 논문 보고 점수 ±15%, 3 seeds 평균. Pong 등 음수 축 환경은 HNS 기준 판정 (문서 내 판정 기준 참조).
- 알고리즘별 구현 노트: `docs/notes/<algo>.md` (구현 중 생성) — 발견한 함정/트릭은 반드시 여기에 기록.

## 현재 진행 상태 (task 완료 시 갱신할 것)

- Task 0 진행 중: `pyproject.toml` / `.python-version` / `uv sync` 완료. `wm/` 패키지, `scripts/`, `tests/`, pre-commit은 미생성 — `docs/superpowers/plans/2026-06-11-task0-repo-bootstrap.md`의 Subtask 2부터 이어서 진행.
- Task 1 완료: `docs/SURVEY.md`, `docs/TARGETS.md` 작성됨 (전 조합 점수·출처 확정, TBD 없음). 잔여는 논문 하이퍼파라미터 → `wm/configs/` yaml 옮기기뿐 (각 알고리즘 task 착수 시 진행하기로 결정). DreamerV2 Atari 100k는 공식 수치 부재로 상대 타깃 채택, DreamerV3는 Nature 2025 수치 채택 (모두 사용자 확인됨). TD-MPC2·DreamerV3-vision @500k는 3자 재실행 테이블(MR.Q) 채택, DreamerV1은 원저자 수치의 CURL 정리본을 **하한 참고치**로 채택 (측정 시점 불일치 — TARGETS.md 노트 참조). Critic 리뷰 반영 완료(2026-06): DMC step 축 = agent step 명문화, 판정은 Pong만 HNS, Task 4는 latent overshooting 채택(사용자 확인됨).

## 절대 규칙

- 레퍼런스 구현 금지: Task 12 이전에는 공식/서드파티 레포(dreamerv3, dreamerv3-torch, SheepRL, TD-MPC2, IRIS, STORM 공식 등)의 코드를 검색·열람·복사하지 않는다. 논문 본문과 부록만 허용. 사용자가 명시적으로 요청한 경우에만 예외.
- PyTorch만 사용: JAX/TF 코드 작성 금지. numpy는 데이터 처리에만. IPEX(아카이브됨)·torch-directml 사용 금지 — XPU는 PyTorch 2.5+ 네이티브 백엔드 사용.
- 공통 모듈 하위 호환: `wm/common/` 수정 시 기존 알고리즘이 깨지지 않아야 한다. 인터페이스 변경이 필요하면 먼저 사용자에게 영향 범위를 보고할 것.
- 노트북 금지: 모든 코드는 `wm/` 패키지 + `scripts/` 엔트리포인트. Colab/Kaggle에서는 git clone 후 실행만.

## 프로젝트 구조 (Task 0에서 생성)

```
wm/common/   envs(atari·dmc 래퍼) · buffer · logger · eval · viz · device · checkpoint
wm/algos/    rssm · dreamer_v1 · dreamer_v2 · dreamer_v3 · tdmpc2 · iris · storm · diamond
wm/configs/  알고리즘×환경 yaml (논문 부록 하이퍼파라미터 기준)
scripts/     train.py · eval.py · collect_random.py · smoke.py 등 엔트리포인트
tests/       pytest
results/     점수 테이블 · 학습 곡선 · open-loop GIF
```

## 환경 / 명령어

- 로컬: Windows (Python 3.11 고정, `.python-version`). uv가 win32 전용으로 resolve하도록 설정됨 — `pyproject.toml`의 `[tool.uv] environments` 수정 금지.
- 패키지 관리: uv. 설치는 `uv sync --extra dev`, 실행은 `uv run python scripts/...`
- 테스트: `uv run pytest` — 공통 모듈 수정 시 반드시 실행
- 린트: `uv run ruff check . && uv run ruff format .` (또는 `uv run pre-commit run --all-files`) — 커밋 전 실행
- 디바이스: `cuda → xpu → mps → cpu` 우선순위 자동 감지 (`wm/common/device.py`). 로컬은 Intel Arc iGPU(xpu), 본 학습 run은 클라우드 CUDA. 디바이스 하드코딩 금지.
- torch/torchvision은 XPU index(`download.pytorch.org/whl/xpu`)에서 설치 — pyproject에 이미 설정됨.

## 코딩 컨벤션

- 모든 학습 스크립트는 `--resume` 지원 필수: 모델·옵티마이저·replay buffer·env step 카운터·RNG 상태를 주기적으로 체크포인트 저장/복원 (`wm/common/checkpoint.py`, Colab 세션 끊김 대비). 상세 설계는 `docs/GPU_CLOUD_SETUP.md`.
- step 축은 env step(= agent step)으로 통일. Atari 100k = 100k agent steps = 400k frames (action repeat 4). DMC 500k = 500k agent steps = 1M frames (action repeat 2). 출처 논문마다 "step" 정의가 다르니 비교 시 `docs/TARGETS.md`의 Step 축 정의 참조.
- 이미지 버퍼 저장은 uint8, 학습 시점에 float 변환.
- config는 hydra-core + omegaconf 기반 yaml. 매직 넘버를 코드에 박지 말고 config로 뺄 것.
- 텐서 shape 주석 필수: `# (B, L, C, H, W)` 형식.
- einops 사용 권장, view/permute 체이닝 지양.
- ruff: line-length 100, py311, `E/F/I` (pyproject 설정 준수).

## 디버깅 원칙 (world model 특화)

RL 점수가 안 나올 때 의심 순서:

1. world model open-loop 예측 품질 (`wm/common/viz.py`의 grid/GIF로 육안 확인)
2. reward/continue head 예측 정확도
3. KL 붕괴 여부 (posterior collapse — KL 항이 0 근처로 떨어지는지)
4. 그 다음에야 actor-critic 쪽 (detach 누락, λ-return, imagination 시작 state의 terminal 포함 여부)

## 작업 방식

- 각 task의 DoD(Definition of Done)를 충족해야 완료. 완료 시 결과물 경로와 메트릭을 요약 보고하고, 이 문서의 "현재 진행 상태"를 갱신.
- 긴 학습 run을 직접 돌리지 말 것 — 학습은 사용자가 Colab/Kaggle/vast.ai에서 실행한다. Claude의 역할은 구현·단위 테스트·짧은 smoke run(수백 step)까지.
- 불확실한 논문 디테일(수식 해석, 하이퍼파라미터 모호함)은 추측으로 구현하지 말고 해석 후보를 제시하고 사용자에게 확인받을 것.
