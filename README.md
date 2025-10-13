# Boundary-Lite Polyp Segmentation (Kvasir-SEG ↔ CVC-ClinicDB)

경량 U-Net-tiny(≈0.118M params) 기반으로 **경계-의식(Region+Boundary) 손실**과 **고정 임계값 평가**를 통해 **도메인 간(내시경 기기/분포) 강건성**을 실험합니다.

---

## Repo 구조

```
boundary-lite-polyp-seg/
├── configs/
│   ├── data.yaml              # 세팅 A/B/C, 경로/이미지 크기
│   └── train_baseline.yaml    # 학습/손실 하이퍼파라미터
├── data/
│   ├── Kvasir-SEG/{images,masks}
│   └── CVC-ClinicDB/{images,masks}
├── outputs/                   # 체크포인트/평가 결과 저장
├── scripts/
│   ├── check_dataset.py       # 데이터 정합성/샘플 오버레이
│   └── eval_test.py           # Test 평가(+샘플 저장, τ 스윕)
├── src/
│   ├── models/unet_tiny.py    # 0.118M 파라미터 U-Net-tiny
│   └── utils/metrics.py       # Dice/IoU 등
└── train.py                   # Baseline/Boundary 학습 스크립트
```

---

## 데이터 준비

본 연구는 두 공개 폴립 세그멘테이션 데이터셋을 사용합니다. 두 데이터셋은 장비/도메인 차이가 있어 크로스-도메인 일반화 평가에 적합합니다.

**1) Kvasir-SEG (≈ 1,000장)**

- 대장내시경 프레임 이미지와 1:1로 매칭되는 바이너리 마스크(폴립=1, 배경=0). 해상도는 다양하며 컬러(RGB) 이미지

- 공식 페이지 & 다운로드: https://datasets.simula.no/kvasir-seg/

(페이지에서 kvasir-seg.zip을 내려받아 압축 해제)


**2) CVC-ClinicDB (≈ 612장)**

- 다른 기관/장비에서 수집된 폴립 이미지+마스크. Kvasir-SEG와 분포 차이가 존재하여 도메인 쉬프트 평가에 사용

- 공식 페이지 & 다운로드: https://polyp.grand-challenge.org/CVCClinicDB/

(페이지 안내에 따라 Original.zip, Ground Truth.zip 다운로드 후 압축 해제)


---

## 학습/평가 기본 커맨드

### 1) 세팅 선택

`configs/data.yaml`:

```yaml
splits:
  setting: A   # A: Train=Kvasir, Test=CVC  | B: Train=CVC, Test=Kvasir
  val_ratio: 0.10
  seed: 42
img_size: [256, 256]
```

### 2) 학습

```bash
python train.py
# 예) device: mps, train 900 / val 100, best val dice ~ 0.56
# 체크포인트: outputs/<out_dir>/best.pt
```

### 3) 테스트 평가 & 샘플 저장

```bash
export PYTHONPATH="$PWD"
python scripts/eval_test.py \
  --ckpt outputs/<실험명>/best.pt \
  --th 0.50 \
  --save_samples 20 \
  --out_dir outputs/test_eval_<TAG>
```

### 4) τ 스윕(임계값 민감도)

```bash
for th in 0.30 0.35 0.40 0.45 0.50 0.55 0.60; do
  od="outputs/test_eval_<TAG>_th${th}"
  python scripts/eval_test.py --ckpt outputs/<실험명>/best.pt \
    --th $th --save_samples 0 --out_dir "$od" >/dev/null
  printf "τ=%-4s  " "$th"; cat "$od/metrics.txt"
done
```

---

## 손실 구성 (Boundary-aware)

최종 손실:

```math
\boxed{
\mathcal{L}
= \lambda_r \bigl(\mathcal{L}_{\text{Dice}}^{\,w} + \mathcal{L}_{\text{BCE}}^{\,w}\bigr)
+ \lambda_b \,\mathcal{L}_{\text{BIoU}}
}
```

```math
\mathcal{L}_{\text{Dice}}^{\,w}
= 1 - 
\frac{2\sum_i w_i\, p_i y_i + \epsilon}
     {\sum_i w_i\, p_i + \sum_i w_i\, y_i + \epsilon},
\qquad
\mathcal{L}_{\text{BCE}}^{\,w}
= -\frac{1}{N}\sum_i w_i \bigl[y_i\log p_i + (1-y_i)\log(1-p_i)\bigr]
```

```math
w_i = 1 + \alpha \exp\!\bigl(- (d_i/\sigma)^2\bigr)
\quad(\text{경계까지의 거리 } d_i,\ \alpha{=}2,\ \sigma{=}3\ \text{예시})
```

```math
\mathcal{L}_{\text{BIoU}}
= 1 - \text{IoU}\bigl(\,\partial_\delta \hat{Y},\ \partial_\delta Y\,\bigr),
\quad
\partial_\delta Y = \text{dilate}(Y,\delta) \setminus \text{erode}(Y,\delta)
\quad(\delta{=}3\ \text{예시})
```

* (p_i): 예측 확률, (y_i): GT 라벨, (w_i): 경계 가중, (\epsilon): 수치 안정화 상수
* (\lambda_r, \lambda_b): 손실 가중 (예: (\lambda_r{=}0.6,\ \lambda_b{=}0.25))

* **Region 가중**: 거리변환 기반 경계 가중 `W_edge = 1 + α exp(-(d/σ)^2)` (기본 `α=2.0, σ=3px`)
* **BIoU(δ=3px)**: 마스크 팽창-침식 경계밴드에서 IoU 최적화
* 기본 하이퍼: `λ_r=0.6, λ_b=0.25` (추가 term들은 향후 Ablation 단계에서 확장)

---

## 실험 로그

### 학습완료 #1 — Baseline (Dice+BCE)

| 실험         | 세팅             |    τ | Test셋 |       Dice |        IoU |   N | 체크포인트                      |
| ---------- | -------------- | ---: | ----- | ---------: | ---------: | --: | -------------------------- |
| Baseline#1 | A (Kvasir→CVC) | 0.50 | CVC   | **0.0068** | **0.0038** | 612 | `outputs/baseline/best.pt` |

보조 정보

* Kvasir-val(100장 샘플 추정) Dice ≈ **0.624** (내-도메인 정상 학습 확인)
* 파라미터 수: **0.118M**

---

### 학습완료 #2 — Boundary-aware (Region+BIoU)

**Run2 (out_dir=outputs/boundary_biou_run2)**
Val(Kvasir) 최고 Dice ≈ **0.56**

| 실험                        | 세팅             |        τ | Test셋 |       Dice |        IoU |   N | 체크포인트                                |
| ------------------------- | -------------- | -------: | ----- | ---------: | ---------: | --: | ------------------------------------ |
| Baseline#2 (run2)         | A (Kvasir→CVC) |     0.50 | CVC   | **0.0293** | **0.0166** | 612 | `outputs/boundary_biou_run2/best.pt` |
| Baseline#2 (run2, best-τ) | A (Kvasir→CVC) | **0.30** | CVC   | **0.0426** | **0.0247** | 612 | ↑ 동일                                 |

동일 가중치 복사 평가(`outputs/boundary_biou_v2/best.pt`)에서도 수치 동일함을 확인.

**요약**

* In-domain(Kvasir-val) 개선은 적절하나, **Cross-domain(CVC)** 성능은 낮음 → **도메인 쉬프트 대응** 필요.

---

## 다음 단계 로드맵

1. **세팅 A 증강 튜닝 (run3)**

   * 밝기/감마/블러/해상도↓: 약/중/강 3단계 합성 쉬프트
   * 색상 정규화 또는 CLAHE 전처리
   * τ 고정 방식: Val에서 F1@τ 최대값으로 선택 → Test에 그대로 적용(운영 시나리오 일치)
2. **세팅 B (Reverse) 평가**

   * Train/Val=CVC → Test=Kvasir; 상호 일반화 확인
3. **캘리브레이션 (추가 예정)**

   * Temperature Scaling, 픽셀-레벨 Reliability diagram / ECE-map
