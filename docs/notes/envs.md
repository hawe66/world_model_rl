# 환경 래퍼 구현 노트 (Task 2)

`wm/common/envs/` — Atari(ALE) · DMC 공통 `Transition` 인터페이스.

## 공통 인터페이스 (`base.py`)

- `Transition(obs, action, reward, is_first, is_last, is_terminal)`.
- `obs`는 dict: `image`(uint8, **HWC**) 또는 `state`(float32, 1-D). float/CHW 변환은 학습 시점.
- **경계 플래그 결정 (사용자 확인됨, 2026-06-15)**:
  - `is_last = terminated or truncated` (에피소드 종료).
  - `is_terminal = terminated`만. **time-limit truncation은 is_terminal=False** → critic 부트스트랩 유지.
  - Dreamer 계열 value 부트스트랩이 time-limit과 진짜 terminal을 구분해야 하므로 채택.

## Atari (`atari.py`)

- base env: `frameskip=1`, `repeat_action_probability=0.0`(**sticky OFF**, Atari 100k 표준). action repeat는 래퍼가 직접 수행.
- 1 agent step = action_repeat(기본 4) raw 프레임. `episode_frame_number`로 검증(테스트).
- **max-pool over last 2 raw frames** → flicker 제거 → resize(BILINEAR) → grayscale(기본) → uint8 HWC `(H,W,1)`, RGB는 `(H,W,3)`.
- **life-loss (기본 OFF)**: 켜면 목숨 감소 시 `is_terminal=True, is_last=False`(에피소드 계속), 진짜 game-over에서만 둘 다 True. 로직은 구현됨, 기본값 False (DreamerV2/Task 6에서 opt-in).
- 함정: `-v5` 환경은 sticky action이 기본 ON(0.25)이라 명시적으로 0.0으로 꺼야 결정성 확보.

## DMC (`dmc.py`)

- proprio(기본): observation dict를 **sorted-key 순서**로 concat → 결정적 `(D,)` float32. walker-walk=24, cheetah-run=17.
- vision: `physics.render(64,64)` → uint8 HWC `(H,W,3)`.
- action repeat 기본 2. `control_steps`로 검증.
- **is_terminal 항상 False**: DMC suite는 고정 길이(1000 control step), 진짜 terminal 없음. is_last는 `TimeStep.last()`(시간초과)에서만 True.
- 에피소드 길이: 1000 control step / action_repeat 2 = 500 agent step.

## 이 개발 머신 한정 함정 (중요)

- **dm_control ↔ mujoco 버전 충돌**: `dm_control 1.0.41`은 `mujoco>=3.8.1`을 선언하지만, mujoco **3.9.0**이 `MjModel.flex_bandwidth` 필드를 제거 → `flex_bandwidth` AttributeError로 모든 DMC 빌드 실패. **`pyproject.toml`에서 `mujoco>=3.8.1,<3.9`로 핀 고정**하여 해결(3.8.1 설치). 사용자 확인됨.
- **DMC vision 렌더링 로컬 불가**: 이 PC는 OpenGL 드라이버가 없어 `physics.render()`가 `GLFWError: WGL ... does not support OpenGL` / `gladLoadGL error`로 실패. 따라서 **proprio는 로컬 완전 테스트, vision은 클라우드(GL 가능)에서 검증**. `tests/test_envs.py`의 vision 테스트는 렌더 불가 시 자동 skip(`_dmc_render_available()`), 렌더 실패로 남는 MuJoCo GL context 소멸자 경고는 `pyproject.toml` `filterwarnings`로 메시지 한정 무시.
- 테스트 실행은 `uv run python -m pytest` (홈드라이브 `c10.dll` 이슈).

## 미결/다음 task로 이월

- env config yaml(`wm/configs/`): 현재 소비자(train script)가 없어 보류. 기본값은 생성자 named kwarg로 노출(매직 넘버 아님). train script 도입(Task 4+) 시 yaml화.
- 결정성 테스트는 proprio/grayscale 기준. vision 결정성은 클라우드에서 추가 검증 권장.
