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

### 3.3 데이터셋 준비

#### Pascal VOC 2012 다운로드

```bash
# VOC 2012 데이터셋 (~2GB) — MMSegmentation 가이드 따름
mkdir -p datasets && cd datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
cd ..
# 압축 해제 후 경로: ./datasets/VOCdevkit/VOC2012/
```

자세한 디렉토리 구조는 [MMSegmentation Data Preparation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html#pascal-voc) 문서 참조.

### 3.4 SAM2 region mask

본 프로젝트의 SR과 MC는 **SAM2가 사전 생성한 region mask**를 입력으로
받습니다. **VOC val 1,449장의 마스크는 본 저장소에 포함되어 있어 별도
다운로드가 불필요**합니다 (`region_masks/voc/`, 약 10MB).

clone 직후 다음 명령으로 확인:

```bash
ls region_masks/voc/ | wc -l    # 1449
```

각 파일은 `{stem}.npz` 형태이고, `instance_mask` 키로 `(H, W)` shape의
`uint8` numpy array를 담고 있습니다 (예: `2007_000033.npz`).

#### 다른 데이터셋에서도 SR/MC를 적용하려면

본 저장소는 VOC val만 마스크를 포함합니다. Pascal Context, COCO 등
다른 데이터셋의 마스크가 필요하면 [`docs/SAM2_MASK_GENERATION.md`](docs/SAM2_MASK_GENERATION.md)
가이드를 따라 직접 생성할 수 있습니다 (별도 SAM2 환경 구축 필요).

### 3.5 데이터셋 경로 설정 (환경변수)

본 저장소의 config는 **환경변수**로 데이터셋 경로를 지정할 수 있도록
설계되어 있습니다. 환경변수 미설정 시 **저장소 기준 상대경로**
(`./datasets/...`)를 자동 사용합니다.

| 환경변수 | 용도 | 기본값 |
|---------|------|--------|
| `VOC_ROOT` | Pascal VOC 위치 | `./datasets/VOCdevkit/VOC2012` |
| `CONTEXT_ROOT` | Pascal Context 위치 | `./datasets/VOC2012` |
| `SAM2_MASK_VOC` | VOC SAM2 마스크 위치 | `./region_masks/voc` (저장소 포함) |

데이터셋이 다른 위치에 있으면 실행 전 `export` 하시면 됩니다:

```bash
export VOC_ROOT=/your/path/to/VOCdevkit/VOC2012
python eval.py --config configs/cfg_voc21.py
```

또는 한 줄로:

```bash
VOC_ROOT=/your/path python eval.py --config configs/cfg_voc21.py
```

> ⚠️ **실행은 반드시 저장소 루트(`NACLIP-SR-MC/`)에서** 해주세요.
> Config의 상대경로는 실행 디렉토리 기준이라, 다른 디렉토리에서 실행하면
> 마스크를 찾지 못해 자동으로 baseline으로 fallback될 수 있습니다.

### 3.6 실행

```bash
conda activate naclip

# Baseline (NACLIP only) — 마스크 미사용
python eval.py --config configs/cfg_voc21_baseline.py

# SR + MC 통합 (본 프로젝트 기여 부분)
python eval.py --config configs/cfg_voc21.py
```

예상 실행 시간 (A100 80GB 기준):
- Baseline: 약 4~6분
- SR + MC: 약 6~8분 (마스크 로드 오버헤드)

---

## 4. 코드 구조

```
NACLIP-SR-MC/
├── README.md                       ← 이 파일
├── LICENSE                         ← MIT (본인 추가분) + Attribution
├── requirements.txt
├── .gitignore
│
├── clip/
│   ├── model.py                    ← ★ SR 통합 (custom_attn 함수)
│   ├── model.py.bak                ← NACLIP 원본 백업
│   └── ... (그 외 NACLIP 원본 파일)
│
├── naclip.py                       ← ★ SAM2 mask 로드 + MC 통합
├── naclip.py.bak                   ← NACLIP 원본 백업
│
├── configs/
│   ├── cfg_voc21.py                ← SR + MC 활성화 (환경변수 기반)
│   ├── cfg_voc21_baseline.py       ← Baseline 검증용 (mask_dir='')
│   ├── cfg_voc20.py, cfg_context*.py, cfg_coco*.py, cfg_ade20k.py, cfg_city_scapes.py
│   │                               ← 다른 벤치마크 config (VOC21만 검증됨)
│   └── ... (cls_*.txt 등)
│
├── region_masks/
│   └── voc/                        ← VOC val SAM2 마스크 1,449장 (저장소 포함)
│
├── eval.py                         ← NACLIP 원본 평가 스크립트
├── visualize.py                    ← 시각화 스크립트
│
├── docs/
│   └── SAM2_MASK_GENERATION.md     ← 다른 데이터셋용 SAM2 마스크 생성 가이드
│
└── vis_results/                    ← 샘플 시각화 출력
```

### 4.1 본 프로젝트가 수정/추가한 파일

| 파일 | 수정 내용 |
|------|---------|
| `clip/model.py` | `custom_attn` 함수에 SR 마스킹 로직 추가 (line 208~220) |
| `naclip.py` | `__init__`에 `mask_dir` 파라미터, `predict`에 mask 로드, `postprocess_result`에 MC 추가 |
| `configs/cfg_voc21.py` | `mask_dir` 활성화 + 환경변수 기반 경로 |
| `configs/cfg_voc21_baseline.py` | `mask_dir=''` (baseline 비교용) |
| `configs/cfg_voc20.py`, `cfg_context*.py` | 환경변수 기반 경로로 정리 |
| `region_masks/voc/` | VOC val SAM2 마스크 1,449장 추가 |
| `docs/SAM2_MASK_GENERATION.md` | 다른 데이터셋용 마스크 생성 가이드 추가 |
| `requirements.txt` | 검증된 패키지 버전 명시 |

원본 파일은 `*.bak` 으로 모두 보존되어 있어 `cp model.py.bak model.py`
한 줄로 NACLIP 순수 baseline으로 즉시 복원 가능.

### 4.2 검증된 범위

| Config | 검증 상태 |
|--------|----------|
| `cfg_voc21.py` | ✅ 본인 환경에서 검증 (VOC21 mIoU = 71.94) |
| `cfg_voc21_baseline.py` | ✅ 본인 환경에서 검증 (VOC21 mIoU = 58.88) |
| `cfg_voc20.py`, `cfg_context*.py` | ⚠️ 절대경로 정리만 됨, 본인 환경에서는 미검증 |
| `cfg_coco*.py`, `cfg_ade20k.py`, `cfg_city_scapes.py` | ⚠️ NACLIP 원본 그대로, 본인 환경에서는 미검증 |

---

## 5. SR / MC 핵심 코드 발췌

### 5.1 Scope Reconstruction (`clip/model.py` 약 208~220 line)

```python
# === Scope Reconstruction (SR) ===
if instance_masks is not None:
    mask_flat = instance_masks[b_idx]   # (P,) per-patch region ID
    # same-region patch pairs → True; different region → False
    E = (mask_flat.unsqueeze(0) == mask_flat.unsqueeze(1))
    # block cross-region attention; CLS row/col preserved
    attn_weights[..., 1:, 1:][~E] = float('-inf')
```

### 5.2 Map Correction (`naclip.py` postprocess_result 부분)

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

샘플 시각화 출력은 [`vis_results/`](vis_results/) 폴더 참조 (6장의 VOC val
이미지).

> 정성적 비교 이미지 (input → baseline → SR+MC → GT) 추가 업로드 예정.

---

## 7. 한계 및 향후 작업

- 추가 벤치마크 평가 진행 중: VOC20, Pascal Context60, COCO-Object
- CorrCLIP의 VR (Value Reconstruction), FR (Feature Refinement) 추가
  통합 시 추가 개선 가능 (논문 보고값 74.8 도달 가능성)
- MC-only ablation은 추가 실험 필요

---

## 8. 캡스톤 디자인 정보

- **소속**: 성균관대학교 Smart Factory Convergence
- **수강 과목**: 캡스톤 디자인
- **수행자**: 김정훈 (hun0919@skku.edu)
- **수행 기간**: 2026년 1학기

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
