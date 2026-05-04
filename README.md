# NACLIP-SR-MC

> CorrCLIP의 **Scope Reconstruction (SR)** 과 **Map Correction (MC)** 모듈을
> NACLIP에 이식하여 training-free Open-Vocabulary Semantic Segmentation
> (OVSS) 성능을 검증한 캡스톤 프로젝트입니다.
>
> Pascal VOC21 기준 mIoU **58.88 → 72.38 (+13.50)** 달성.

---

## 1. 프로젝트 개요

본 저장소는 다음 두 논문의 결합 실험 코드를 공개합니다.

- **베이스 모델**: NACLIP (Hajimiri et al., WACV 2025)
- **이식 모듈**: CorrCLIP (Zhang et al., ICCV 2025 Oral) 의 SR + MC

> 원본 NACLIP 코드는 `https://github.com/sinahmr/NACLIP` 에서 가져왔으며,
> 본 저장소의 모든 변경 사항은 SR + MC 통합과 관련된 최소한의 수정입니다.
> 자세한 attribution은 `LICENSE` 파일과 본 README의 9. References 절 참고.

### 1.1 동기

지도교수님으로부터 받은 캡스톤 디자인 과제는 **Open Vocabulary Semantic
Segmentation 분야의 SOTA 개선**입니다. 본 프로젝트는 ICCV 2025 Oral인
CorrCLIP을 분석한 뒤, 그 핵심 컴포넌트가 다른 training-free 모델에서도
유효한지 검증하기 위해 NACLIP에 이식하는 실험을 수행했습니다.

### 1.2 왜 NACLIP에 이식했는가

| 후보 | 결과 | 비고 |
|------|------|------|
| DeOP (training-based, 1차 시도) | ❌ 실패 | GPS-CAL 구조적 종속성으로 attention 변경 불가 |
| **NACLIP (training-free, 2차 시도)** | ✅ 성공 | ClearCLIP과 동일한 reduced 구조 → SR과 호환 |

DeOP에서 실패한 분석 자체도 학술적 가치가 있어 캡스톤 보고서에 포함되어
있습니다. 본 저장소는 **성공한 2차 시도(NACLIP + SR + MC)** 만을 다룹니다.

---

## 2. 핵심 결과

### Pascal VOC21 (1,449 images, 21 classes)

| Setting | mIoU | Δ vs baseline |
|---------|------|---------------|
| NACLIP (baseline 재현) | **58.88** | — (논문 보고값 58.8과 일치) |
| + SR (Scope Reconstruction) | 59.77 | +0.89 |
| + SR + MC | **71.94** | **+13.06** |
| + SR + MC + PAMR | **72.38** | **+13.50** |

**참고**: CorrCLIP 원본 (full SR + VR + MC + FR, ViT-B/16) = 74.8 →
본 저장소의 SR + MC 두 모듈만으로 원본 성능의 **96.7%** 도달.

### 통합 모듈 동작 원리 (간단 요약)

- **Scope Reconstruction (SR)** — Attention 단계에서 SAM2 region mask를
  활용하여 **다른 region에 속한 patch 간의 attention을 `-inf`로 마스킹**.
  Inter-class correlation 자체를 차단.

- **Map Correction (MC)** — Segmentation 결과 후처리 단계에서 SAM2
  region 내부의 모든 픽셀을 **majority voting (`torch.mode`)** 으로 통일.
  공간적 일관성을 강제.

---

## 3. 빠른 시작 (Quick Start)

### 3.1 필수 환경

| 항목 | 검증된 값 |
|------|----------|
| GPU | NVIDIA A100 80GB (24GB 이상 권장) |
| OS | Ubuntu 20.04 / 22.04 (Linux container) |
| Python | 3.10 |
| CUDA | 11.8 |
| PyTorch | 2.0.0+cu118 |
| mmcv | 2.0.1 |
| mmsegmentation | 1.1.1 |

### 3.2 설치

```bash
# 1) 저장소 clone
git clone https://github.com/KimJH0919/NACLIP-SR-MC.git
cd NACLIP-SR-MC

# 2) Conda 환경 생성
conda create -n naclip python=3.10 -y
conda activate naclip

# 3) PyTorch (CUDA 11.8 빌드)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# 4) mmcv (PyTorch 버전 일치하는 pre-built wheel 사용)
pip install mmcv==2.0.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html

# 5) 나머지 의존성
pip install -r requirements.txt
```

### 3.3 데이터셋 / 마스크 준비

#### (a) Pascal VOC 2012 다운로드

```bash
# VOC 2012 데이터셋 (~2GB) — MMSegmentation 가이드 따름
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
# 압축 해제 후 경로: ./VOCdevkit/VOC2012/
```

자세한 디렉토리 구조는 [MMSegmentation Data Preparation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html#pascal-voc) 문서 참조.

#### (b) SAM2 region mask 준비

본 프로젝트의 SR과 MC는 **사전 생성된 SAM2 region mask가 필요합니다**
(VOC val 1,449 장의 `.npz` 파일).

마스크 생성 방법은 두 가지:

```bash
# 옵션 1: 본인이 직접 생성 (권장)
# - SAM2 환경 별도 구축 후 scripts/generate_sam2_masks.py 실행
# - 설치 가이드: https://github.com/facebookresearch/sam2
# - 자세한 절차는 docs/SAM2_MASK_GENERATION.md 참조

# 옵션 2: 사전 생성된 마스크 다운로드 (제공 예정)
# - Google Drive 링크: [업로드 후 갱신]
```

마스크는 `~/region_masks/voc/{stem}.npz` 형태로 저장되며, 각 파일은
`instance_mask` 키로 `(H, W)` shape의 `uint8` numpy array를 담고 있습니다.

### 3.4 Config 경로 설정

`configs/cfg_voc21.py` 에서 두 경로를 본인 환경에 맞게 수정:

```python
data_root = '/path/to/VOCdevkit/VOC2012'   # Pascal VOC 위치
mask_dir  = '/path/to/region_masks/voc'    # SAM2 mask 위치
```

### 3.5 실행

```bash
conda activate naclip

# Baseline (NACLIP only) — 마스크 미사용
python eval.py --config configs/cfg_voc21_baseline.py

# SR + MC 통합 (본 프로젝트 기여 부분)
python eval.py --config configs/cfg_voc21.py
```

예상 실행 시간 (A100 80GB 기준):
- Baseline: 약 4 ~ 6분
- SR + MC: 약 6 ~ 8분 (마스크 로드 오버헤드)

---

## 4. 코드 구조

```
NACLIP-SR-MC/
├── README.md                     ← 이 파일
├── LICENSE                       ← MIT (본인 추가분) + Attribution
├── requirements.txt
├── .gitignore
│
├── clip/
│   ├── model.py                  ← ★ SR 통합 (custom_attn 함수)
│   ├── model.py.bak              ← NACLIP 원본 백업
│   └── ... (그 외 NACLIP 원본 파일)
│
├── naclip.py                     ← ★ SAM2 mask 로드 + MC 통합
├── naclip.py.bak                 ← NACLIP 원본 백업
│
├── configs/
│   ├── cfg_voc21.py              ← SR + MC 활성화
│   ├── cfg_voc21_baseline.py     ← Baseline 검증용
│   └── ... (다른 데이터셋 config)
│
├── eval.py                       ← NACLIP 원본
├── visualize.py                  ← 시각화 스크립트 (수정됨)
│
├── docs/
│   ├── SAM2_MASK_GENERATION.md   ← SAM2 마스크 생성 가이드
│   ├── figures/                  ← 파이프라인 다이어그램
│   └── sample_results/           ← 정성적 비교 이미지 (입력/baseline/SR+MC/GT)
│
└── scripts/
    └── generate_sam2_masks.py    ← (선택) SAM2 마스크 일괄 생성
```

### 4.1 본 프로젝트가 수정/추가한 파일

| 파일 | 수정 내용 |
|------|---------|
| `clip/model.py` | `custom_attn` 함수에 SR 마스킹 로직 추가 (line 208~220) |
| `naclip.py` | `__init__`에 `mask_dir` 파라미터, `predict`에 mask 로드, `postprocess_result`에 MC 추가 |
| `configs/cfg_voc21.py` | `mask_dir` 활성화 |
| `configs/cfg_voc21_baseline.py` | `mask_dir=''` (baseline 비교용) |

원본 파일은 `*.bak` 으로 모두 보존되어 있어 `cp model.py.bak model.py`
한 줄로 NACLIP 순수 baseline으로 즉시 복원 가능.

---

## 5. SR / MC 핵심 코드 발췌

### 5.1 Scope Reconstruction (`clip/model.py:208~220`)

```python
# === Scope Reconstruction (SR) ===
if instance_masks is not None:
    mask_flat = instance_masks[b_idx]   # (P,) per-patch region ID
    # same-region patch pairs → True; different region → False
    E = (mask_flat.unsqueeze(0) == mask_flat.unsqueeze(1))
    # block cross-region attention; CLS row/col preserved
    attn_weights[..., 1:, 1:][~E] = float('-inf')
```

### 5.2 Map Correction (`naclip.py:postprocess_result`)

```python
# === Map Correction (MC) ===
for region_id in torch.unique(sam2_mask):
    if region_id == 0:
        continue                          # skip unsegmented region
    region = (sam2_mask == region_id)
    seg_pred[region] = torch.mode(seg_pred[region])[0]
```

---

## 6. 실행 결과 화면

자세한 결과는 [`docs/sample_results/`](docs/sample_results/) 참조.

> ⚠️ 정성적 비교 이미지(input → baseline → SR+MC → GT)는 별도 업로드
> 예정입니다. 본 README는 검증된 정량적 결과만 포함합니다.

---

## 7. 한계 및 향후 작업

- 추가 벤치마크 평가 진행 중: VOC20, Pascal Context60, COCO-Object
- CorrCLIP의 VR (Value Reconstruction), FR (Feature Refinement) 추가
  통합 시 추가 개선 가능 (논문 보고값 74.8 → 96.7% 도달 중)
- MC-only ablation은 추가 실험 필요

---

## 8. 캡스톤 디자인 정보

- **소속**: 성균관대학교 Smart Factory Convergence
- **수강 과목**: 캡스톤 디자인
- **수행자**: 김정훈 (hun0919@skku.edu)
- **수행 기간**: 2025년 1학기

---

## 9. References

본 프로젝트는 다음 논문/저장소를 기반으로 합니다.

```bibtex
@inproceedings{naclip_wacv2025,
  title  = {Pay Attention to Your Neighbours: Training-Free Open-Vocabulary Semantic Segmentation},
  author = {Hajimiri, Sina and Ben Ayed, Ismail and Dolz, Jose},
  booktitle = {WACV},
  year   = {2025}
}

@inproceedings{corrclip_iccv2025,
  title  = {CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation},
  author = {Zhang, Dengke and Liu, Fagui and Tang, Quan},
  booktitle = {ICCV (Oral)},
  year   = {2025}
}

@inproceedings{sclip_eccv2024,
  title  = {SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference},
  author = {Wang, Feng and Mei, Jieru and Yuille, Alan},
  booktitle = {ECCV},
  year   = {2024}
}

@inproceedings{clip_icml2021,
  title  = {Learning Transferable Visual Models From Natural Language Supervision},
  author = {Radford, Alec and others},
  booktitle = {ICML},
  year   = {2021}
}

@article{sam2_2024,
  title  = {SAM 2: Segment Anything in Images and Videos},
  author = {Ravi, Nikhila and others},
  journal= {arXiv:2408.00714},
  year   = {2024}
}
```

원본 저장소:
- NACLIP: https://github.com/sinahmr/NACLIP
- CorrCLIP: https://github.com/zdk258/CorrCLIP
- SAM2: https://github.com/facebookresearch/sam2
- MMSegmentation: https://github.com/open-mmlab/mmsegmentation

---

## 10. 라이선스 / Attribution

자세한 내용은 [`LICENSE`](LICENSE) 파일 참고.

요약:
- **본 저장소의 추가/수정 코드**: MIT License (Copyright 김정훈, 2025)
- **NACLIP 원본 코드**: 원본 저장소 (`sinahmr/NACLIP`) 의 저작권 보유.
  upstream에 명시된 라이선스가 없어 학술적 fair-use 관행으로 사용. 본
  README와 `LICENSE` 파일에서 명확한 attribution을 제공함.
- **CorrCLIP**: SR/MC의 알고리즘 출처. 본 저장소는 알고리즘 재구현이며
  해당 논문(`Zhang et al., ICCV 2025`) 인용 필수.

---

## 11. AI 도구 사용 명시

본 프로젝트의 코드 분석, 문서 작성, 실험 설계 검토 과정에서 다음 AI 도구를
활용하였습니다.

> Anthropic. (2025), Claude Opus 4 (claude-opus-4-20250514),
> https://claude.ai/

활용 범위: NACLIP과 CorrCLIP 코드 비교 분석, 통합 지점 결정, 결과 해석,
README/보고서 작성 보조.
