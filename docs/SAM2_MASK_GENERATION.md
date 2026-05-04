# SAM2 Region Mask 생성 가이드

본 프로젝트의 SR (Scope Reconstruction) 과 MC (Map Correction) 모듈은 SAM2가
사전 생성한 region mask를 입력으로 받습니다. 이 문서는 Pascal VOC val 1,449장
에 대한 SAM2 mask 생성 절차를 설명합니다.

## 검증된 환경

- 별도 conda 환경 권장 (이름: `sam2env`) — 메인 `naclip` 환경과 분리
- GPU: NVIDIA A100 80GB (16GB 이상이면 충분, fp16 사용)
- 실행 시간: VOC 1,449장 기준 약 5~10분

## 1. SAM2 환경 설치

```bash
# 새 conda 환경
conda create -n sam2env python=3.10 -y
conda activate sam2env

# PyTorch (CUDA 11.8)
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
    -f https://download.pytorch.org/whl/cu118/torch_stable.html

# SAM2 설치 (공식 저장소)
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .

# 추가 의존성
pip install iopath
```

## 2. SAM2 모델 가중치 다운로드

```bash
# Hiera-L 가중치 (~860MB)
mkdir -p ~/sam2_checkpoints
cd ~/sam2_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## 3. VOC val 1,449장에 대한 mask 생성

```python
# scripts/generate_sam2_masks_voc.py
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# SAM2 import
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# 경로 설정 (본인 환경에 맞게 조정)
VOC_VAL_LIST = '/path/to/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
VOC_IMG_DIR  = '/path/to/VOCdevkit/VOC2012/JPEGImages'
SAM2_CKPT    = '/path/to/sam2_hiera_large.pt'
SAM2_CFG     = 'sam2_hiera_l.yaml'    # SAM2 repo 내장
OUT_DIR      = '/path/to/region_masks/voc'

os.makedirs(OUT_DIR, exist_ok=True)

# 모델 로드 (fp16)
sam2_model = build_sam2(
    SAM2_CFG, SAM2_CKPT,
    device='cuda', apply_postprocessing=False,
).half()
mask_gen = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.7,
    multimask_output=False,
)

# val 이미지 목록
with open(VOC_VAL_LIST) as f:
    stems = [line.strip() for line in f if line.strip()]

print(f"Generating masks for {len(stems)} VOC val images...")
for stem in tqdm(stems):
    out_path = os.path.join(OUT_DIR, f'{stem}.npz')
    if os.path.exists(out_path):
        continue

    img_path = os.path.join(VOC_IMG_DIR, f'{stem}.jpg')
    img = np.array(Image.open(img_path).convert('RGB'))

    with torch.autocast('cuda', dtype=torch.float16):
        masks = mask_gen.generate(img)

    # region map 구성: 작은 region부터 큰 region 순으로 라벨링 (큰 게 위에 덮음)
    H, W = img.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)
    for idx, ann in enumerate(sorted(masks, key=lambda x: x['area']), start=1):
        out[ann['segmentation']] = min(idx, 255)
        # 0 = unsegmented, 1~254 = region IDs

    np.savez_compressed(out_path, instance_mask=out)

print(f"\nAll masks saved to {OUT_DIR}")
print(f"File count: {len(os.listdir(OUT_DIR))}")
```

## 4. 검증

```bash
# 마스크 파일 개수 확인 (1,449장이어야 함)
ls /path/to/region_masks/voc/ | wc -l

# 한 개 샘플 확인
python -c "
import numpy as np
m = np.load('/path/to/region_masks/voc/2007_000033.npz')['instance_mask']
print(f'shape: {m.shape}, dtype: {m.dtype}')
print(f'unique region IDs: {len(np.unique(m))} (예: 18~25 정도)')
print(f'min/max: {m.min()}/{m.max()}')
"
```

기대되는 출력 (이미지마다 다름):
```
shape: (375, 500), dtype: uint8
unique region IDs: 18 (예: 18~25 정도)
min/max: 0/17
```

## 5. 다른 데이터셋 (Pascal Context, COCO 등)

위 스크립트의 경로 부분만 수정하여 동일하게 적용 가능. 출력 경로 예시:

| 데이터셋 | 이미지 수 | 권장 출력 경로 |
|---------|-----------|-------------|
| VOC val | 1,449 | `region_masks/voc/` |
| Pascal Context val | 5,104 | `region_masks/context/` |
| COCO val | 5,000 | `region_masks/coco/` |
| ADE20K val | 2,000 | `region_masks/ade/` |
| Cityscapes val | 500 | `region_masks/city/` |

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-----------|
| `ModuleNotFoundError: iopath` | `pip install iopath` |
| OOM (out of memory) | `points_per_side=16` 으로 축소 |
| 너무 느림 | fp16 적용 확인 (`.half()` + `autocast`) |
| 일부 region이 누락됨 | `pred_iou_thresh`, `stability_score_thresh` 를 0.5~0.6으로 낮춤 |
