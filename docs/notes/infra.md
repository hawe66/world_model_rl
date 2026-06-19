# 공통 인프라 구현 노트 (Task 3)

`wm/common/`의 buffer · logger · eval · viz. 알고리즘 코드가 학습 로직에만 집중하도록 하는 공유 인프라. RL/월드모델 로직은 미포함(Task 4+).

## buffer (`buffer.py`) — `SequenceReplayBuffer`

- **샘플링 방식 결정 (사용자 확인됨, 2026-06-18): 경계 허용 + is_first 리셋 (DreamerV2/V3)**.
  - 버퍼를 삽입 순서의 연속 스트림으로 보고, 전체 유효 범위 `[0, size-L]`에서 uniform start → 길이 L 청크. 청크가 **에피소드 경계를 넘을 수 있음**.
  - 청크 내 새 에피소드 시작에 `is_first=True` → 소비자(RSSM)가 그 지점에서 recurrent state 리셋. 후보 B(경계 내 제한) 대비 짧은 에피소드 처리·데이터 낭비 없음이 이유.
- 링 버퍼: 물리 배열 capacity 고정, `_start`(가장 오래된 것의 물리 인덱스)로 logical→physical 매핑(modulo). 가득 차면 FIFO eviction. **logical 인덱싱이 곧 삽입 순서**라 wrap 경계에서 데이터 불연속 없음.
- 이미지 uint8 저장, `sample(..., to_float=True)`면 float32 [0,1]. CHW 변환·device 이동은 소비자 몫.
- 반환은 numpy 배치(`(B, L, ...)`); torch/device는 Task 4 trainer에서. (`numpy는 데이터 처리용` 규칙 준수)
- `state_dict`/`load_state_dict`: logical 순서로 compact 스냅샷(origin 0 리셋) → `checkpoint.py`의 `replay_buffer` 인자와 왕복 검증됨(--resume).
- `from_npz`/`load_npz`: Task 2 collect_random 스키마(image|state·action·reward·is_first/last/terminal·episode) 로딩.

## logger (`logger.py`)

- TensorBoard(`SummaryWriter`) + `metrics.jsonl` 이중 기록. step 축 = env step.
- `log_scalar`는 한 호출당 `{"step": s, "<key>": v}` 한 줄 append. context manager 지원.

## eval (`eval.py`)

- `evaluate(env, policy, episodes, *, mode, max_steps)` → `{return_mean, return_std, returns, length_mean, lengths}`.
- **eval env는 호출자가 train env와 분리해서 넘기는 계약**(별도 인스턴스/seed). 함수는 분리 자체를 강제하지 않고 분리된 env를 받는다.
- `policy(obs, mode)` 호출 인터페이스. 랜덤 policy로 단위 검증 가능(월드모델/actor 불필요).

## viz (`viz.py`)

- `frames_to_grid`(필름스트립/그리드 PNG), `frames_to_gif`(GIF), `open_loop_grid(real, pred)`(real 위 / pred 아래).
- 모델 부재 → `open_loop_grid`는 인터페이스만; 데모/테스트는 `pred=real`. Task 4 RSSM 디코더 출력이 `pred`로 들어옴.
- grayscale(C=1)은 PIL/GIF 저장 시 채널 squeeze.
- **함정**: imageio v2 GIF에서 `fps` 키워드는 deprecated → `duration=1000/fps`(프레임당 ms) 사용으로 교체.

## 데모

- `scripts/demo_buffer_viz.py --npz <npz> --seq-len --batch --out`: npz→버퍼→`(B,L)` 샘플→grid/gif/open_loop 저장. Pong npz(61 step, 단일 에피소드)로 `(4,16,64,64,1)` 배치 확인.
- proprio-only npz는 이미지가 없어 시각화 불가(데모가 명시적으로 안내하고 종료).

## 이월

- hydra config 배선(버퍼 용량·L·batch 등)은 Task 4 train script에서. 현재는 전부 함수/생성자 인자(매직 넘버 아님).
