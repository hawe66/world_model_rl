# GPU 클라우드 설정 가이드

> 로컬 개발(Windows/WSL)에서 코드를 작성하고, GPU 학습은 외부 클라우드에서 실행하는 워크플로우.
> 핵심 원칙: **노트북 셀에 코드를 짜지 말 것.** Colab/Kaggle은 실행 환경일 뿐이다.

---

## 1. 클라우드 티어별 현실적 평가

| 옵션 | 무료 GPU | 세션 제한 | 주간 한도 | 비용 | 용도 |
|---|---|---|---|---|---|
| Google Colab (무료) | T4 (15GB) | ~12시간 | 비공식 제한 있음 | 무료 | 연기 확인, 빠른 프로토타입 |
| Kaggle Notebooks | T4/P100 (16GB) | 12시간 | **주 30시간** | 무료 | 단일 run 완주, 병행 활용 |
| Google Colab Pro | A100 (40GB) | 24시간+ | 컴퓨팅 유닛 제한 | ~$10/월 | DreamerV3 multi-seed |
| vast.ai | RTX 4090 등 | 없음 | 없음 | $0.2~0.6/h | 본 run 단계 권장 |
| RunPod | RTX 4090 등 | 없음 | 없음 | $0.3~0.5/h | vast.ai 대안 |

### 무료 티어의 한계

- DreamerV3 단일 run이 6~12시간 → Colab 12시간 세션 경계선에 걸림
- 3 seeds × 3 게임 × 6~12시간 = 54~108시간 → 무료 티어로는 **절대 전량 재현 불가**
- **권장 전략**: 알고리즘 1 seed 빠른 검증은 Kaggle, 전체 재현 run은 vast.ai/RunPod

---

## 2. 워크플로우 원칙

```
[로컬 Windows]                    [클라우드 GPU]
코드 작성/커밋                         |
      |                               |
      └── git push ──────────────→ !git clone
                                  !pip install -e ".[dev]"
                                  !python scripts/train.py \
                                      --config wm/configs/dreamer_v3/atari.yaml \
                                      --env Pong --seed 0 \
                                      --resume  # 체크포인트 있으면 자동 재개
```

- **모든 학습 코드는 `scripts/` 아래 Python 파일**로 유지
- Colab/Kaggle 노트북은 `git clone` + `python scripts/train.py` 셀 2~3개만
- 결과는 Google Drive 또는 Kaggle Output으로 저장 → 로컬에서 pull

---

## 3. 체크포인트 / 재개 설계 (필수)

DreamerV3 run이 6시간 이상인데 Colab 세션은 12시간 전후로 끊긴다.
**`--resume` 플래그를 처음부터 설계해야 한다.**

### 저장 대상

```python
checkpoint = {
    "step": env_step,                    # env step (학습 진행도)
    "model": world_model.state_dict(),
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
    "model_opt": model_optimizer.state_dict(),
    "actor_opt": actor_optimizer.state_dict(),
    "critic_opt": critic_optimizer.state_dict(),
    "replay_buffer": buffer.state_dict(),  # 버퍼 재활용 (선택)
    "rng_state": torch.get_rng_state(),
    "numpy_rng": np.random.get_state(),
}
```

### 저장 주기 및 경로

```python
# 권장 저장 전략
CKPT_EVERY_STEPS = 50_000          # 50k env step마다 저장 (DreamerV3 run 기준 ~30분)
CKPT_DIR = Path(cfg.run_dir) / "checkpoints"
CKPT_LATEST = CKPT_DIR / "latest.pt"
CKPT_BEST = CKPT_DIR / "best.pt"   # eval 점수 기준 best 별도 보관
```

### Google Drive 동기화 (Colab)

```python
# Colab 셀 상단에서 실행
from google.colab import drive
drive.mount('/content/drive')

RUN_DIR = "/content/drive/MyDrive/world_model_zoo/dreamer_v3/pong_seed0"
```

```bash
# train.py 실행 시 run_dir를 Drive 경로로 지정
!python scripts/train.py \
    --config wm/configs/dreamer_v3/atari.yaml \
    --env Pong --seed 0 \
    --run-dir "/content/drive/MyDrive/world_model_zoo/dreamer_v3/pong_seed0" \
    --resume
```

### Kaggle Output 동기화

```bash
# Kaggle은 /kaggle/working/ 이 Output으로 자동 저장됨
!python scripts/train.py \
    --config wm/configs/dreamer_v3/atari.yaml \
    --env Pong --seed 0 \
    --run-dir "/kaggle/working/dreamer_v3_pong_seed0" \
    --resume
```

### `--resume` 플래그 구현 스케치

```python
# scripts/train.py 또는 wm/common/trainer.py
def load_checkpoint(run_dir: Path, device: str) -> dict | None:
    ckpt_path = run_dir / "checkpoints" / "latest.pt"
    if ckpt_path.exists():
        print(f"[resume] Loading checkpoint from {ckpt_path}")
        return torch.load(ckpt_path, map_location=device)
    return None

def save_checkpoint(state: dict, run_dir: Path, is_best: bool = False):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "latest.pt"
    torch.save(state, path)
    if is_best:
        shutil.copy(path, ckpt_dir / "best.pt")
```

---

## 4. 표준 Colab 노트북 구조

```python
# [셀 1] 환경 설정
!git clone https://github.com/<user>/world-model-zoo.git
%cd world-model-zoo
!pip install uv -q
!uv sync --extra dev -q

# [셀 2] Drive 마운트 (선택)
from google.colab import drive
drive.mount('/content/drive')
RUN_DIR = "/content/drive/MyDrive/world_model_zoo/dreamer_v3/pong_seed0"

# [셀 3] 학습 실행
!python scripts/train.py \
    --config wm/configs/dreamer_v3/atari.yaml \
    --env Pong --seed 0 \
    --run-dir "$RUN_DIR" \
    --resume
```

---

## 5. 표준 Kaggle 노트북 구조

```python
# [셀 1] 환경 설정
import subprocess
subprocess.run(["git", "clone", "https://github.com/<user>/world-model-zoo.git"], check=True)
subprocess.run(["pip", "install", "uv", "-q"], check=True)
subprocess.run(["uv", "sync", "--extra", "dev", "-q"],
               cwd="/kaggle/working/world-model-zoo", check=True)

# [셀 2] 학습 실행 (세션 12시간 제한 → --resume 필수)
subprocess.run([
    "python", "scripts/train.py",
    "--config", "wm/configs/dreamer_v3/atari.yaml",
    "--env", "Pong", "--seed", "0",
    "--run-dir", "/kaggle/working/dreamer_v3_pong_seed0",
    "--resume"
], cwd="/kaggle/working/world-model-zoo", check=True)
```

---

## 6. vast.ai / RunPod 사용 (본 run 권장)

```bash
# vast.ai: RTX 4090, PyTorch 2.5 이미지 선택
# SSH 접속 후:
git clone https://github.com/<user>/world-model-zoo.git
cd world-model-zoo
pip install uv
uv sync --extra dev

# 3 seeds 병렬 실행 (GPU 메모리 허용 시)
for seed in 0 1 2; do
    python scripts/train.py \
        --config wm/configs/dreamer_v3/atari.yaml \
        --env Pong --seed $seed \
        --run-dir runs/dreamer_v3/pong_seed${seed} &
done
wait

# 결과 로컬 복사
rsync -avz <user>@<vast-ip>:/root/world-model-zoo/runs/ ./runs/
```

---

## 7. XPU (Intel Arc) 로컬 학습

이 프로젝트의 `pyproject.toml`은 PyTorch XPU 인덱스를 기본 소스로 설정한다.
Intel Arc GPU가 없거나 CUDA를 쓸 때는 아래와 같이 소스를 교체한다.

```toml
# pyproject.toml — CUDA 전환 시
[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

코드 내 device 선택은 `wm/common/device.py`의 유틸 함수를 통해 추상화:

```python
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```
