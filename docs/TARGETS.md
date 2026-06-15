# 타깃 점수 테이블 (Task 1 산출물)

> 전 프로젝트의 재현 성공 기준선. **논문 보고 점수 ±15%, 3 seeds 평균** (로드맵 원칙).
> 평가 프로토콜: 학습 중 10 episodes 평균, 최종 점수는 마지막 체크포인트 기준 30 episodes 평균.
> 모든 점수는 논문 원문 테이블에서 추출. 원문에 없는 조합은 출처와 함께 근사/보류로 표기.

## Step 축 정의 (사용자 확인됨)

- 프로젝트의 env step = **agent step** (CLAUDE.md Atari 관례와 동일: Atari 100k = 100k agent steps = 400k frames @AR4).
- DMC 500k = **500k agent steps = 1M frames** (AR2 기준).
- 주의: 출처 논문마다 "step"의 정의가 다르다 — CURL은 simulator **frame** 기준, MR.Q는 **agent step** 기준, DreamerV3 원문 "environment steps"는 **frame** 기준 (Table 9가 Atari 100k를 "400K environment steps"로 표기함에서 확인). 아래 테이블은 모든 행에 양쪽 환산을 병기한다.

## 판정 기준

- 기본: 논문 raw 점수의 ±15% 구간 진입.
- **예외는 Pong 하나**: 점수 축이 음수를 포함(−21~21)해 raw ±15%가 무의미 → HNS 기준 판정.
  `HNS = (score − random) / (human − random)`, 타깃 HNS의 ±15% 구간. 그 외 환경은 전부 raw ±15% (단순화 — Breakout은 random=1.7이라 HNS 특례의 실익 없음).
- Atari random/human 기준치 (DIAMOND Table 1과 동일): Pong −20.7/14.6, Breakout 1.7/30.5, Boxing 0.1/12.1.

## 1. Atari 100k (100k agent steps = 400k frames, sticky action off)

| 알고리즘 | Pong | Breakout | Boxing | 출처 (원문) | 교차 확인 |
|---|---|---|---|---|---|
| DreamerV2 | — | — | — | **공식 100k 수치 없음** (아래 노트) | — |
| DreamerV3 (arXiv v1, 2023) | 18 | 31 | 78 | DreamerV3 arXiv:2301.04104**v1** Table S.1 "Atari scores at 400K environment frames" (ar5iv v1 확인) | DIAMOND Tab.1 (18.0/31.0/78.0) ✓ · HarmonyDream Tab.3 "Original" ✓ |
| **DreamerV3 (Nature 2025) ← 채택 타깃** | **−4** | **10** | **82** | arXiv:2301.04104**v2** **Table 9** "Atari100k scores" (직접 확인) | — (v1과 상이, 아래 노트) |
| IRIS | 14.6 | 83.7 | 70.1 | IRIS arXiv:2209.00588 Table 1 (직접 확인) | DreamerV3 v2 Table 9 (15/84/70) ✓ · DIAMOND Tab.1 ✓ |
| STORM | 11.3 | 15.9 | 79.7 | STORM arXiv:2310.09615 Table 2 (PDF 직접 확인 — 원문 표기는 정수 11/16/80) | DIAMOND Tab.1 (11.3/15.9/79.7) ✓ |
| DIAMOND (5 seeds) | 20.4 | 132.5 | 86.9 | DIAMOND arXiv:2405.12399 Table 1 (직접 확인) | — |
| (참고) TWM | 18.8 | 20.0 | 77.5 | TWM arXiv:2303.07109 Table 1 (ar5iv 확인, 5 runs/game) | DIAMOND Tab.1 · DreamerV3 v2 Table 9 ✓ |

### ±15% 타깃 구간 (채택 수치 기준)

| 알고리즘 | Pong (HNS 판정) | Breakout (raw) | Boxing (raw) |
|---|---|---|---|
| DreamerV3 (Nature) | HNS 0.40~0.54 → raw **−6.5~−1.5** | **8.5~11.5** | **69.7~94.3** |
| IRIS | 9.3~19.9 (HNS 0.85~1.15) | 71.1~96.3 | 59.6~80.6 |
| STORM | 6.5~16.1 (HNS 0.77~1.04) | 13.5~18.3 | 67.7~91.7 |
| DIAMOND | 14.2~26.6 (HNS 0.99~1.34) | 112.6~152.4 | 73.9~99.9 |

판정 규칙: Pong만 HNS(음수 축), Breakout·Boxing은 raw ±15% (위 판정 기준 참조).

### 노트 — DreamerV2 (Task 6)

원논문(arXiv:2010.02193)은 200M frames 기준이며, 100k per-game 수치는 TWM·IRIS·STORM·DIAMOND·DreamerV3·HarmonyDream 어느 baseline 테이블에도 없음 (2026-06 조사). **결정(사용자 확인됨): Task 6은 절대 점수 대신 상대 타깃** —

1. 동일 조건(우리 구현, Atari 100k)에서 **V2 > V1 계열 성능** 및 **V2 < V3** 순위 재현
2. 참고 기준선: 동급 categorical-latent 계열인 TWM 100k 점수 (Pong 18.8 / Breakout 20.0 / Boxing 77.5)
3. 절대 점수는 기록만 하고 ±15% 판정 제외

### 노트 — DreamerV3 버전 차이

arXiv v1(2023)과 Nature 2025(arXiv v2)의 Atari 100k 수치가 크게 다름 (Pong 18→−4, Breakout 31→10, Boxing 78→82; v2는 gamer mean 125%). **결정(사용자 확인됨): 구현 기준과 동일한 Nature 2025 수치를 타깃**으로 한다 (판정은 위 판정 기준대로 Pong만 HNS, Breakout·Boxing은 raw ±15%). v1 수치는 참고용으로 병기.

### 노트 — STORM 원문 직접 열람 (해소됨)

STORM 원문 PDF 테이블 직접 확인 완료 (Random/Human/SimPLe/TWM/IRIS/DreamerV3/STORM 컬럼, Human Mean 126.7%). 원문은 정수 표기(Pong 11, Breakout 16, Boxing 80)이며, 소수점 수치(11.3/15.9/79.7)는 DIAMOND Table 1 기재값 채택 — 양자 정합.

## 2. DMC (프로젝트 예산: 500k agent steps = 1M frames @AR2)

| 알고리즘 | 환경 (입력) | 점수 | 출처의 측정 시점 (agent steps / frames) | 출처 | 지위 |
|---|---|---|---|---|---|
| DreamerV1 | walker-walk (vision) | **897 ± 49** | **250k** / 500k (CURL "500k"는 frame 기준, AR2) | CURL arXiv:2004.04136 Table 1 (DMControl500k, Dreamer 컬럼) | **하한 참고치** (사용자 확인됨 — 아래 노트) |
| DreamerV1 | cheetah-run (vision) | **570 ± 253** | **125k** / 500k (CURL은 cheetah에 **AR4** 사용) | 상동 | **하한 참고치** (AR4 vs 우리 AR2 불일치 — 아래 노트) |
| DreamerV3 | walker-walk (vision) | **942** [936, 949] | **500k** / 1M — 프로젝트 예산과 동일 | MR.Q(Fujimoto et al., ICLR 2025, arXiv:2501.16142) **Table 6** "DMC - Visual final results" (10 seeds 재실행) | **±15% 판정 타깃** |
| DreamerV3 (참고) | walker-walk (proprio) | **936** | 원문 "500K env steps"는 frame 기준 → frames 500k (AR2 가정 시 agent 250k) | arXiv:2301.04104v2 Table 11 (직접 확인) | 참고용 |
| DreamerV3 (참고) | cheetah-run (proprio) | **614** | 상동 | 상동 Table 11 | 참고용 |
| TD-MPC2 | walker-walk (proprio) | **981** [979, 984] | **500k** / 1M — 프로젝트 예산과 동일 | MR.Q **Table 5** "DMC - Proprioceptive final results" (10 seeds 재실행; ULD arXiv:2602.12643 Table 3 동일 수치 확인) | **±15% 판정 타깃** (3자 재실행 — 아래 노트) |
| TD-MPC2 | cheetah-run (proprio) | **917** [915, 920] | 상동 | 상동 | **±15% 판정 타깃** (3자 재실행 — 아래 노트) |
| TD-MPC2 (stretch) | walker-walk (vision) | **958** [952, 965] | **500k** / 1M | MR.Q **Table 6** | 참고용 (Task 8 본문 기준 vision은 stretch — 로드맵 고정 벤치마크 표와의 관계는 아래 노트) |

### ±15% 타깃 구간 (판정 대상만)

| 알고리즘 | walker-walk | cheetah-run |
|---|---|---|
| DreamerV3 (vision) | 801~1000 | — |
| TD-MPC2 (proprio) | 834~1000 | 779~1000 |

DreamerV1은 ±15% 판정 제외 — 아래 노트의 하한 조건으로 대체. DreamerV3 proprio(936/614)는 참고용이라 판정 없음.

### 노트 — DreamerV1: 출처 성격과 하한 참고치 강등 (사용자 확인됨)

- **출처 성격**: CURL Table 1의 Dreamer 컬럼은 재실행이 아니라 **원저자(Hafner et al.) 보고 결과를 CURL이 표로 정리한 것** ("We chose Dreamer because the authors report performance for all of the above environments"). 캡션의 "10 seeds"는 CURL 자기 점수에 대한 것으로 Dreamer 컬럼엔 비적용 — DreamerV1 원문 실험은 **5 seeds**.
- **측정 시점 불일치**: CURL의 "500k"는 frame 기준 → walker-walk는 agent step **250k**, cheetah-run은 CURL이 **AR4**를 쓰므로 agent step **125k** 시점. 프로젝트 예산(500k agent steps)의 1/2~1/4이라 ±15% 판정에 쓰면 정상 구현이 구간을 "초과"하는 역설 발생.
- **판정 방식 (결정)**: 500k agent step 시점 점수가 **CURL 기재치 이상**이면 통과(하한 조건). 학습 공정성은 곡선 모양(단조 상승, CURL 시점 환산 위치에서의 근접도)으로 함께 확인.
- **AR 불일치**: cheetah-run은 CURL이 AR4, 우리 Task 2 계획은 AR2 — 동일 frame 예산이어도 agent step 수·탐사 특성이 달라 직접 비교 불가함을 유의.
- 로드맵 고정 벤치마크 표가 DreamerV1을 "DMC (proprio)"로 적은 것은 오기 — Task 5 본문과 TARGETS 모두 **vision이 정본** (로드맵 표 수정 제안 별도).

### 노트 — TD-MPC2·DreamerV3 vision 출처

TD-MPC2 원문(arXiv:2310.16828)은 per-task 수치 테이블 없이 집계·곡선만 제공 (PDF 직접 확인). 따라서 MR.Q Table 5·6의 500k agent step 시점 10-seeds **재실행** 테이블을 채택 (이쪽은 진짜 재실행 — CURL의 "원저자 수치 정리"와 성격이 다름). ULD(arXiv:2602.12643) Table 3에서 동일 수치 재확인됨. 측정 시점이 프로젝트 예산과 정확히 일치하므로 ±15% 판정 타깃으로 적합.

DreamerV3 vision walker-walk: MR.Q 942 [936,949] ↔ 원문 Table 12의 961은 **동일 예산 비교** — 원문 "1M environment steps"는 frame 기준이라 1M frames = 500k agent steps로 MR.Q의 측정 시점과 같다 (이전 판의 "조기 수렴" 해석은 단위 혼동에 의한 오류, 철회). 차이(~2%)는 재실행 변동 수준. 반면 proprio Table 11의 936/614는 frames 500k(= MR.Q 예산의 절반) 시점이라 MR.Q의 898/699와는 **예산이 다른 비교**임 — 둘 다 기재만 하고 동일성 주장 안 함.

## 3. 알고리즘×환경 매핑 요약 (로드맵 고정 벤치마크)

| | Pong | Breakout | Boxing | walker-walk (v) | walker-walk (p) | cheetah-run (v) | cheetah-run (p) |
|---|---|---|---|---|---|---|---|
| DreamerV1 (Task 5) | | | | 897(하한) | | 570(하한) | |
| DreamerV2 (Task 6) | 상대 | 상대 | 상대 | | | | |
| DreamerV3 (Task 7) | −4* | 10* | 82 | 942 | 936(참고) | | 614(참고) |
| TD-MPC2 (Task 8) | | | | 958(stretch) | 981 | | 917 |
| IRIS (Task 9) | 14.6 | 83.7 | 70.1 | | | | |
| STORM (Task 10) | 11.3 | 15.9 | 79.7 | | | | |
| DIAMOND (Task 11) | 20.4 | 132.5 | 86.9 | | | | |

\* Pong은 HNS 기준 판정 (그 외 raw ±15%). (v)=vision, (p)=proprio. "(하한)"은 ±15% 대신 하한 조건 판정.
