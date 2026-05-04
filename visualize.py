"""
NACLIP + SR + MC 시각화
선택한 이미지에 대해 segmentation 결과를 overlay하여 저장
"""
import os, sys, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, '.')
import clip
from clip.model import VisionTransformer

# VOC21 클래스명 + 색상
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
               'sofa', 'train', 'tvmonitor']

# VOC 색상 팔레트
def voc_colormap(N=256):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b, j = 0, 0, 0, i
        for _ in range(8):
            r |= ((j >> 0) & 1) << (7 - _)
            g |= ((j >> 1) & 1) << (7 - _)
            b |= ((j >> 2) & 1) << (7 - _)
            j >>= 3
        cmap[i] = [r, g, b]
    return cmap

def visualize_one(img_path, mask_dir, net, text_features, query_idx, device,
                  prob_thd=0.1, logit_scale=40, save_path='vis.png'):
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize

    # Load and preprocess image
    orig_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = orig_img.size

    transform = Compose([
        Resize((336, 336), interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
    ])
    img_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Load SAM2 mask
    stem = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(mask_dir, f'{stem}.npz')
    instance_masks = None
    instance_masks_orig = None

    if os.path.exists(mask_path):
        mask = np.load(mask_path)['instance_mask']
        mask_tensor = torch.from_numpy(mask.astype(np.int32))
        instance_masks_orig = mask_tensor.clone()

        # Downsample for SR
        mask_down = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=(21, 21), mode='nearest'
        ).squeeze().long()
        instance_masks = mask_down.view(1, -1).to(device)

    # Forward
    with torch.no_grad():
        features = net.encode_image(img_tensor, return_all=True,
                                     instance_masks=instance_masks)
        features = features[:, 1:]
        features = features / features.norm(dim=-1, keepdim=True)
        logits = features @ text_features.T

    # Reshape to spatial
    patch_size = net.visual.patch_size
    h_p = img_tensor.shape[2] // patch_size
    w_p = img_tensor.shape[3] // patch_size
    logits = logits.permute(0, 2, 1).reshape(1, -1, h_p, w_p)

    # Upsample to original size
    logits = F.interpolate(logits, size=(orig_h, orig_w),
                           mode='bilinear', align_corners=False)

    # Classification
    seg_probs = torch.softmax(logits[0] * logit_scale, dim=0)
    num_cls = max(query_idx) + 1
    num_queries = len(query_idx)
    if num_cls != num_queries:
        query_idx_t = torch.tensor(query_idx, dtype=torch.int64)
        cls_index = F.one_hot(query_idx_t).T
        cls_index = cls_index.view(num_cls, num_queries, 1, 1).to(device)
        seg_probs = (seg_probs.unsqueeze(0) * cls_index).max(1)[0]

    seg_pred = seg_probs.argmax(0)

    # Map Correction
    if instance_masks_orig is not None:
        mc_mask = F.interpolate(
            instance_masks_orig.unsqueeze(0).unsqueeze(0).float(),
            size=(orig_h, orig_w), mode='nearest'
        ).squeeze().long().to(device)
        mask_values = torch.unique(mc_mask)
        mask_values = mask_values[mask_values != 0]
        for mv in mask_values:
            region = (mv == mc_mask)
            if region.sum() > 0:
                seg_pred[region] = torch.mode(seg_pred[region])[0]

    seg_pred = seg_pred.cpu().numpy()

    # --- Also compute baseline (no SR, no MC) for comparison ---
    with torch.no_grad():
        features_base = net.encode_image(img_tensor, return_all=True,
                                          instance_masks=None)
        features_base = features_base[:, 1:]
        features_base = features_base / features_base.norm(dim=-1, keepdim=True)
        logits_base = features_base @ text_features.T

    logits_base = logits_base.permute(0, 2, 1).reshape(1, -1, h_p, w_p)
    logits_base = F.interpolate(logits_base, size=(orig_h, orig_w),
                                mode='bilinear', align_corners=False)
    seg_probs_base = torch.softmax(logits_base[0] * logit_scale, dim=0)
    if num_cls != num_queries:
        seg_probs_base = (seg_probs_base.unsqueeze(0) * cls_index).max(1)[0]
    seg_pred_base = seg_probs_base.argmax(0).cpu().numpy()

    # Load GT if available
    gt_path = img_path.replace('JPEGImages', 'SegmentationClass').replace('.jpg', '.png')
    gt = None
    if os.path.exists(gt_path):
        gt = np.array(Image.open(gt_path))
        gt[gt == 255] = 0  # ignore → background

    # Visualize
    cmap = voc_colormap()
    n_cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Original image
    axes[0].imshow(np.array(orig_img))
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Baseline
    seg_color_base = cmap[seg_pred_base]
    axes[1].imshow(seg_color_base)
    axes[1].set_title(f'NACLIP (baseline)')
    axes[1].axis('off')

    # Ours
    seg_color = cmap[seg_pred]
    axes[2].imshow(seg_color)
    axes[2].set_title(f'NACLIP + SR + MC (Ours)')
    axes[2].axis('off')

    # GT
    if gt is not None:
        gt_color = cmap[gt]
        axes[3].imshow(gt_color)
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')

    # Legend
    unique_classes = np.unique(np.concatenate([seg_pred.flatten(),
                                               seg_pred_base.flatten()]))
    if gt is not None:
        unique_classes = np.unique(np.concatenate([unique_classes,
                                                    gt[gt < 21].flatten()]))
    patches = [mpatches.Patch(color=cmap[c]/255., label=VOC_CLASSES[c])
               for c in unique_classes if c < 21]
    fig.legend(handles=patches, loc='lower center', ncol=min(7, len(patches)),
               fontsize=9, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def main():
    device = torch.device('cuda')

    # Load model
    net, _ = clip.load('ViT-B/16', device=device, jit=False)
    net.visual.set_params('reduced', 'naclip', 5.)
    net.eval()

    # Load text features (VOC21)
    from prompts.imagenet_template import openai_imagenet_template
    with open('./configs/cls_voc21.txt') as f:
        name_sets = f.readlines()

    class_names, query_idx = [], []
    for idx, line in enumerate(name_sets):
        names = line.strip().split(', ')
        class_names += names
        query_idx += [idx] * len(names)

    text_features = []
    with torch.no_grad():
        for name in class_names:
            tokens = clip.tokenize([t(name) for t in openai_imagenet_template]).to(device)
            feat = net.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.mean(0)
            feat = feat / feat.norm()
            text_features.append(feat)
    text_features = torch.stack(text_features)

    # Select images
    voc_root = os.environ.get('VOC_ROOT', './datasets/VOCdevkit/VOC2012')
    mask_dir = os.environ.get('SAM2_MASK_VOC', './region_masks/voc')
    img_dir = os.path.join(voc_root, 'JPEGImages')

    # Choose diverse images
    val_file = os.path.join(voc_root, 'ImageSets/Segmentation/val.txt')
    with open(val_file) as f:
        val_ids = [l.strip() for l in f.readlines()]

    # Pick 6 images
    selected = val_ids[:6]

    os.makedirs('./vis_results', exist_ok=True)
    for stem in selected:
        img_path = os.path.join(img_dir, f'{stem}.jpg')
        save_path = f'./vis_results/{stem}.png'
        print(f'Processing {stem}...')
        visualize_one(img_path, mask_dir, net, text_features, query_idx,
                      device, save_path=save_path)

    print(f'\nDone! {len(selected)} images saved to ./vis_results/')


if __name__ == '__main__':
    main()
