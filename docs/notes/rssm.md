# RSSM (PlaNet) 구현 노트 (Task 4)

`wm/algos/rssm/` — Dreamer 계보의 코어. RL 없이 순수 시퀀스 모델링으로 검증. dreamer_v1~v3의 공유 베이스.

## 아키텍처 (PlaNet, 사용자 확인 hparam)

- model state = `[h_t (deter), z_t (stoch)]` concat. decoder·reward head 입력은 이 결합 state (stochastic 단독 아님 — SURVEY 주의 반영).
- `h_t = GRU(h_{t-1}, MLP([z_{t-1}, a_{t-1}]))`, prior `p(z|h)`, posterior `q(z|h,e)` 모두 Gaussian(softplus std + min_std 0.1).
- 차원(recall, 부록 확인 대상): deter=200, stoch=30, hidden=200, enc 채널 [32,64,128,256] k4 s2 → 1024 embed, decoder 대칭 transposed(k 5,5,6,6), reward MLP 2×200. Adam lr 1e-3 eps 1e-4, grad-clip 1000, B=50 L=50.
- **CNN은 64×64 고정** (PlaNet). 다른 해상도면 enc/dec 커널 재계산 필요.

## 손실 결정 (사용자 확인됨, 2026-06-18)

- recon NLL = unit-variance Gaussian → `0.5·Σ_pixel (x-μ)²`, reward NLL = `0.5·(r-μ)²`.
- **Latent overshooting (Full)**: 각 t·거리 d∈[1,D]마다 posterior@t에서 prior를 d-step 굴려 `KL(sg(q[t+d]) ‖ prior_d)`, 1/D 평균, ×β. **posterior 타깃에 stop-grad**. is_first를 만나면 그 roll 중단(에피소드 경계 안 넘음). D=config(`overshoot_d`, 기본 seq_len).
- **free nats**: step별 latent-합산 KL에 `clamp(min=free_nats=3)`. clamp 하한이라 KL이 임계 미만이면 grad 0(free-nats 본래 동작).

## ★ 버퍼 리셋 계약 (가장 중요한 함정)

`observe()`는 두 규칙을 명시 구현:
1. **t=0 init**: 모든 윈도우가 `initial(B)`(h=0,z=0)에서 시작. 중간 시작 윈도우(is_first[:,0]=False)도 init됨 — 버퍼가 윈도우 시작에 is_first를 강제로 켜주지 않으므로(`docs/notes/infra.md`) 이 무조건 init이 필수. is_first만 믿으면 init 신호를 놓침.
2. **is_first 추가 리셋**: `keep=(1-is_first[:,t])`로 prev state·prev action을 0으로 마스킹 → 경계에서 recurrent state 리셋.
- 단위 테스트 `test_is_first_resets_deterministic_state`: 리셋 지점의 deter가 리셋 이전 history에 무관(동일), 비리셋 시 history에 의존(상이)임을 검증.
- action 인덱싱: `action[:,t]`는 obs[t]로 이끈 전이 action(우리 envs 컨벤션). 그래서 h_t 계산에 `action[:,t]` 사용이 맞음. is_first(리셋) 지점의 dummy action(0)은 마스킹으로도 0 보장.

## 안정성 관찰 (DreamerV2 복선용)

- **KL balancing 부재(의도적)**: 여기선 free nats까지만, prior/posterior 비대칭 가중 없음. 본 결정의 직접 귀결 — overshooting KL이 **sg(posterior)** 를 타깃으로 하므로 posterior는 주로 recon/reward로 학습되고 prior가 posterior로 끌려간다. posterior collapse(KL→0) 또는 prior 부정확이 장기 학습에서 보이면 → DreamerV2의 KL balancing(예: 0.8/0.2)이 해법. **Task 6에서 도입 예정, 여기선 증상만 관찰.**
- **smoke 관찰(pong, 10 step, tiny)**: loss 438→365 단조 감소, recon 감소, reward_mse 감소, **kl 0.16→0.27로 상승(0 붕괴 아님)**, overshoot=3.0(=free_nats, 초기 prior≈sg-posterior라 clamp 하한). 본 검증은 클라우드 walker-walk vision에서 (DoD 4~6).

## 로컬/클라우드 분리

- 로컬(CPU): 구현·단위 테스트·`--smoke`(수십 step, tiny dims). pong.npz(grayscale C=1, discrete→one-hot)로 동작 확인.
- 클라우드(GPU): walker-walk **vision**(GL 필요) 본 학습. open-loop 5-step context→45-step prior 전개 GIF에서 운동 20 step↑ 유지(DoD 핵심), reward MSE 단조 감소, KL 비붕괴 판정.
- `train_rssm.py`: image C·action(discrete one-hot vs continuous)을 npz에서 자동 감지. config는 `wm/configs/rssm/default.yaml`(+`smoke` 섹션 override). `--resume`로 model·optim·**replay_buffer**·step·RNG 복원.

## 이월

- 부록 hparam 정확값 확인 시 default.yaml·이 노트 갱신.
- KL balancing은 Task 6(DreamerV2)에서.
