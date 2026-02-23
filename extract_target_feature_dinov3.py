import cv2
import torch
import numpy as np
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
# Ensure sane DPI and fonts to avoid FreeType invalid ppem issues
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 12
import matplotlib.patches as patches
from dinov3_feature_extractor import DINOv3FeatureExtractor
import os
import time
import subprocess
from pathlib import Path
import torch.nn.functional as F

class ROISelector:
    """A class to handle ROI selection with matplotlib."""
    def __init__(self, ax):
        self.ax = ax
        self.rect = None
        self.start_point = None
        self.end_point = None
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (event.xdata, event.ydata)
        if self.rect:
            self.rect.remove()
        self.rect = patches.Rectangle(self.start_point, 0, 0, linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.rect)

    def on_release(self, event):
        if event.inaxes != self.ax or self.start_point is None:
            return
        self.end_point = (event.xdata, event.ydata)
        width = self.end_point[0] - self.start_point[0]
        height = self.end_point[1] - self.start_point[1]
        self.rect.set_width(width)
        self.rect.set_height(height)
        self.ax.figure.canvas.draw()

    def get_bbox(self):
        if self.start_point and self.end_point:
            x1 = int(min(self.start_point[0], self.end_point[0]))
            y1 = int(min(self.start_point[1], self.end_point[1]))
            x2 = int(max(self.start_point[0], self.end_point[0]))
            y2 = int(max(self.start_point[1], self.end_point[1]))
            return [x1, y1, x2, y2]
        return None

# =====================
# IPG helpers (optional)
# =====================
def _run_ipg_inference(ipg_root: Path, ipg_python: Path, ckpt_dir: Path, pose_dir: Path, ref_dir: Path, out_dir: Path):
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["HF_HUB_ENABLE_XET"] = "0"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    cmd = [
        str(ipg_python),
        "inference.py",
        "--ckpt_dir", str(ckpt_dir),
        "--pose_dir", str(pose_dir),
        "--ref_dir", str(ref_dir),
        "--out_dir", str(out_dir),
    ]
    subprocess.run(cmd, cwd=str(ipg_root), env=env, check=True)


def _crop_generated_from_canvas(canvas_path: Path, num_poses: int, w: int = 128, h: int = 256):
    from PIL import Image
    img = Image.open(str(canvas_path)).convert("RGB")
    crops = []
    for i in range(num_poses):
        x1 = (2 * i + 2) * w
        y1 = 0
        x2 = x1 + w
        y2 = h
        if x2 <= img.width and y2 <= img.height:
            crops.append(img.crop((x1, y1, x2, y2)))
        else:
            break
    return crops

# 计算IoU的辅助函数
def compute_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def main(args):
    """Main function to extract and save target features from a folder of images using DINOv3."""
    
    image_folder = args.image_folder
    if not os.path.isdir(image_folder):
        print(f"Error: Provided path '{image_folder}' is not a valid directory.")
        return

    image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print(f"Error: No valid image files found in '{image_folder}'.")
        return

    all_features = []

    print("Initializing DINOv3 Feature Extractor...")
    try:
        extractor = DINOv3FeatureExtractor(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            verbose=False
        )
    except Exception as e:
        print(f"Failed to initialize DINOv3 feature extractor: {e}")
        print("Please ensure DINOv3 models are downloaded and paths are correct.")
        return

    # 初始化 YOLOE（可选）
    yolo_model = None
    if args.yolo_model and os.path.exists(args.yolo_model):
        try:
            print(f"Initializing YOLOE model: {args.yolo_model}")
            from ultralytics import YOLO
            yolo_model = YOLO(args.yolo_model)
            if args.classes:
                try:
                    text_pe = yolo_model.get_text_pe(args.classes)
                    yolo_model.set_classes(args.classes, text_pe)
                    print(f"YOLOE classes set via prompts: {args.classes}")
                except AttributeError:
                    print("Warning: Current ultralytics version does not support YOLOE text prompts; proceeding without prompts.")
        except Exception as e:
            print(f"Failed to initialize YOLOE: {e}")
            yolo_model = None
    else:
        if args.yolo_model:
            print(f"Warning: YOLO model path not found: {args.yolo_model}. Proceeding without YOLOE.")

    for i, image_path in enumerate(image_files):
        print(f"\n--- Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not load image from {image_path}. Skipping.")
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, figsize=(10, 8))
        # Normalize tk scaling to avoid invalid ppem font sizing on some systems
        try:
            fig.canvas.get_tk_widget().tk.call('tk', 'scaling', 1.0)
        except Exception:
            pass
        ax.imshow(image_rgb)
        ax.set_title("Draw a box around the target. Close the window to confirm.")
        
        roi_selector = ROISelector(ax)
        
        print("A window has popped up. Please draw a bounding box around the target.")
        print("Close the window when you are done.")
        plt.show()
        
        bbox = roi_selector.get_bbox()
        
        if bbox is None:
            print("No bounding box was selected for this image. Skipping.")
            continue

        bbox_xyxy = np.array([bbox])
        print(f"Selected BBox [x1, y1, x2, y2]: {bbox_xyxy[0]}")
        
        # 基于 YOLOE 的实例分割掩码（如可用）
        det_bbox = bbox_xyxy[0].tolist()
        mask_bool = None
        if yolo_model is not None:
            try:
                results = yolo_model.predict(image_bgr, conf=args.confidence)
                if results and results[0].boxes:
                    r = results[0]
                    boxes = r.boxes
                    masks_np = None
                    if getattr(r, 'masks', None) is not None and getattr(r.masks, 'data', None) is not None:
                        masks_tensor = r.masks.data
                        masks_np = (masks_tensor > 0.5).detach().cpu().numpy().astype('uint8')
                        # 对齐到原图尺寸
                        H, W = image_bgr.shape[:2]
                        resized_masks = []
                        for m in masks_np:
                            if m.shape[0] != H or m.shape[1] != W:
                                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                            resized_masks.append(m.astype('uint8'))
                        masks_np = resized_masks
                    # 选择与用户框 IoU 最大的检测
                    best_idx, best_iou = -1, 0.0
                    for idx, box in enumerate(boxes):
                        xyxy_det = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        iou = compute_iou(bbox_xyxy[0].tolist(), xyxy_det)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    if best_idx >= 0:
                        det_bbox = boxes[best_idx].xyxy[0].cpu().numpy().astype(int).tolist()
                        if masks_np is not None and best_idx < len(masks_np):
                            mask_bool = masks_np[best_idx].astype(bool)
                        print(f"YOLOE matched detection index: {best_idx}, IoU: {best_iou:.3f}")
            except Exception as e:
                print(f"YOLOE inference failed, fallback to ROI mask. Error: {e}")
                mask_bool = None

        # 若未得到实例掩码，则使用矩形ROI作为掩码
        if mask_bool is None:
            H, W = image_bgr.shape[:2]
            mask_bool = np.zeros((H, W), dtype=bool)
            x1, y1, x2, y2 = bbox_xyxy[0]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 > x1 and y2 > y1:
                mask_bool[y1:y2, x1:x2] = True
            print("Using rectangular ROI as mask.")

        # 基于掩码提取特征：将背景置零，仅保留掩码内像素
        masked_img = image_bgr.copy()
        masked_img[~mask_bool] = 0

        # 可视化掩码叠加（可选）
        if getattr(args, 'show_mask', False):
            overlay_rgb = image_rgb.copy()
            alpha = 0.35
            color = np.array([255, 0, 0], dtype=np.float32)  # 红色 (RGB)
            overlay_rgb[mask_bool] = (
                overlay_rgb[mask_bool].astype(np.float32) * (1.0 - alpha) + color * alpha
            ).astype(np.uint8)
            fig2, ax2 = plt.subplots(1, figsize=(10, 8))
            try:
                fig2.canvas.get_tk_widget().tk.call('tk', 'scaling', 1.0)
            except Exception:
                pass
            ax2.imshow(overlay_rgb)
            x1, y1, x2, y2 = det_bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lime', facecolor='none')
            ax2.add_patch(rect)
            ax2.set_title("Mask Preview (close this window to continue)")
            plt.show()

        print("Extracting base DINOv3 feature (masked ROI)...")
        feature = extractor.extract_features(masked_img, np.array([det_bbox]))  # [1, C]
        f_base = F.normalize(feature[0], dim=0)  # [C]

        # 数据增强：基于掩码的水平/竖直翻转与小角度旋转
        f_list = [f_base]

        # 水平翻转（启用 IPG 时跳过）
        if not getattr(args, "use_ipg_for_person", False):
            masked_img_h = cv2.flip(masked_img, 1)
            mask_h = np.fliplr(mask_bool)
            ys, xs = np.where(mask_h)
            if len(xs) and len(ys):
                hx1, hy1, hx2, hy2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                f_h = extractor.extract_features(masked_img_h, np.array([[hx1, hy1, hx2, hy2]]))
                f_list.append(F.normalize(f_h[0], dim=0))
            else:
                print("[FlipAug] Skipped hflip due to empty mask after flip.")

        # 竖直翻转（可选）
        if getattr(args, "aug_vflip", False):
            masked_img_v = cv2.flip(masked_img, 0)
            mask_v = np.flipud(mask_bool)
            ys, xs = np.where(mask_v)
            if len(xs) and len(ys):
                vx1, vy1, vx2, vy2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                f_v = extractor.extract_features(masked_img_v, np.array([[vx1, vy1, vx2, vy2]]))
                f_list.append(F.normalize(f_v[0], dim=0))
            else:
                print("[FlipAug] Skipped vflip due to empty mask after flip.")

        # 小角度旋转（可选，±deg）
        deg = float(getattr(args, "aug_rotate_deg", 0.0) or 0.0)
        if deg > 0.0:
            H_img, W_img = mask_bool.shape[:2]
            center = (W_img / 2.0, H_img / 2.0)
            for ang in (-deg, deg):
                M = cv2.getRotationMatrix2D(center, ang, 1.0)
                rot_img = cv2.warpAffine(
                    masked_img, M, (W_img, H_img),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
                )
                rot_mask_u8 = cv2.warpAffine(
                    (mask_bool.astype(np.uint8) * 255), M, (W_img, H_img),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                rot_mask = rot_mask_u8 > 0
                ys, xs = np.where(rot_mask)
                if len(xs) and len(ys):
                    rx1, ry1, rx2, ry2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                    f_r = extractor.extract_features(rot_img, np.array([[rx1, ry1, rx2, ry2]]))
                    f_list.append(F.normalize(f_r[0], dim=0))
                else:
                    print(f"[RotAug] Skipped rotation {ang} due to empty mask after rotate.")

        f_base_aug_all = F.normalize(torch.stack(f_list, dim=0).mean(dim=0), dim=0)

        # Optional: IPG-based centralization for person
        if args.use_ipg_for_person:
            try:
                base_dir = Path(__file__).parent.resolve()
                ipg_root = Path(args.ipg_root)
                if not ipg_root.is_absolute():
                    ipg_root = (base_dir / ipg_root).resolve()
                ckpt_dir = Path(args.ipg_ckpt_dir) if args.ipg_ckpt_dir else (ipg_root / "pretrained")
                pose_dir = Path(args.ipg_pose_dir) if args.ipg_pose_dir else (ipg_root / "standard_poses")
                ipg_python = Path(args.ipg_python) if args.ipg_python else (ipg_root / ".venv-ipg/bin/python")

                ts = int(time.time())
                tmp_ref_dir = ipg_root / f"tmp_ref_{ts}"
                tmp_out_dir = ipg_root / f"tmp_out_{ts}"
                tmp_ref_dir.mkdir(parents=True, exist_ok=True)
                tmp_out_dir.mkdir(parents=True, exist_ok=True)

                use_pose_dir = pose_dir

                # Save ROI crop as ref image for IPG
                x1, y1, x2, y2 = bbox
                roi_bgr = image_bgr[y1:y2, x1:x2]
                if roi_bgr.size == 0:
                    raise RuntimeError("Empty ROI crop for IPG")
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                ref_name = "ref.jpg"
                ref_path = tmp_ref_dir / ref_name
                from PIL import Image
                Image.fromarray(roi_rgb).save(ref_path)

                print("Running IPG generation (this may take a while)...")
                _run_ipg_inference(ipg_root, ipg_python, ckpt_dir, use_pose_dir, tmp_ref_dir, tmp_out_dir)

                canvas_path = tmp_out_dir / ref_name
                if not canvas_path.exists():
                    raise RuntimeError(f"IPG output canvas not found: {canvas_path}")

                # Crop generated person images from canvas and extract DINO features
                gen_pils = _crop_generated_from_canvas(canvas_path, args.ipg_num_poses, w=128, h=256)
                f_gen_list = []
                for pil_img in gen_pils:
                    # 原生成图
                    gen_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    h2, w2 = gen_bgr.shape[:2]
                    f_i = extractor.extract_features(gen_bgr, np.array([[0, 0, w2, h2]]))  # [1, C]
                    f_gen_list.append(F.normalize(f_i[0], dim=0))
                    # 水平翻转增强（对生成图）
                    if getattr(args, 'ipg_flip_aug', False):
                        try:
                            pil_h = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                            gen_bgr_h = cv2.cvtColor(np.array(pil_h), cv2.COLOR_RGB2BGR)
                            hh, ww = gen_bgr_h.shape[:2]
                            f_h = extractor.extract_features(gen_bgr_h, np.array([[0, 0, ww, hh]]))
                            f_gen_list.append(F.normalize(f_h[0], dim=0))
                        except Exception:
                            pass

                # 额外：对原始 ROI 做一次水平翻转增强（与生成集一并融合）
                if getattr(args, 'ipg_flip_aug', False):
                    try:
                        roi_bgr_h = cv2.flip(roi_bgr, 1)
                        h0, w0 = roi_bgr_h.shape[:2]
                        f_ori_h = extractor.extract_features(roi_bgr_h, np.array([[0, 0, w0, h0]]))
                        f_gen_list.append(F.normalize(f_ori_h[0], dim=0))
                    except Exception:
                        pass

                if f_gen_list:
                    f_gen_mean = torch.stack(f_gen_list, dim=0).mean(dim=0)
                    f_fused = F.normalize(f_base_aug_all + args.ipg_eta * f_gen_mean, dim=0)
                else:
                    print("[IPG] No generated crops found, fallback to base-aug feature.")
                    f_fused = f_base_aug_all

                all_features.append(f_fused.unsqueeze(0))  # keep [1, C] item semantics
                print(f"DINOv3 fused feature (with flip/rot-aug) ready. Dim: {f_fused.numel()}")

            except Exception as e:
                print(f"[IPG] Augmentation failed: {e}. Fallback to base-aug feature.")
                all_features.append(f_base_aug_all.unsqueeze(0))
            finally:
                # Clean temporary artifacts silently (best-effort)
                try:
                    import shutil
                    shutil.rmtree(tmp_ref_dir, ignore_errors=True)
                    shutil.rmtree(tmp_out_dir, ignore_errors=True)
                except Exception:
                    pass
        else:
            # No IPG: keep base masked feature with flip/rot augmentation
            all_features.append(f_base_aug_all.unsqueeze(0))
            print(f"DINOv3 base feature (with flip/rot-aug) ready. Dim: {f_base_aug_all.numel()}")

    if not all_features:
        print("No features were extracted. Exiting.")
        return

    # --- Feature Set Creation ---
    print(f"\n--- Creating a DINOv3 feature set with {len(all_features)} features ---")
    feature_set = [f.squeeze(0) for f in all_features]

    # --- Save the feature set ---
    output_path = args.output_path
    if os.path.isdir(output_path):
        print(f"Output path '{output_path}' is a directory. Saving to 'feature_set_dinov3.pt' inside it.")
        output_path = os.path.join(output_path, "feature_set_dinov3.pt")
    
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(feature_set, output_path)
    print(f"\nDINOv3 feature set with {len(feature_set)} features saved successfully to {output_path}")
    if feature_set:
        print(f"Shape of the first feature in the set: {feature_set[0].shape}")
        print(f"Feature dimension: {feature_set[0].numel()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save a fused DINOv3 feature for a target object from a folder of images.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing input images.")
    parser.add_argument("--output_path", type=str, default="target_feature_dinov3.pt", help="Path to save the output fused feature tensor.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation ('cuda' or 'cpu').")
    parser.add_argument("--model_name", type=str, default="dinov3_vith16plus", help="DINOv3 model name (dinov3_vith16plus, dinov3_vitl16, etc.)")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="dinov3-main/dinov3/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
                        help="Path to DINOv3 checkpoint file.")
    # 新增：YOLOE 分割用于掩码提取
    parser.add_argument("--yolo_model", type=str, default="ORTrack/weights/yoloe-11l-seg.pt", help="YOLOE segmentation weights path.")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLOE detection confidence threshold.")
    parser.add_argument("--classes", type=str, nargs='+', help="Optional class prompts for YOLOE (e.g., person car).")
    parser.add_argument("--show_mask", action='store_true', help="Show mask overlay preview during feature extraction.")
    # IPG centralization (optional, for person targets)
    parser.add_argument("--use_ipg_for_person", action='store_true', help="Use IPG-based centralization (person only) to fuse a single enhanced feature.")
    parser.add_argument("--ipg_flip_aug", action='store_true', default=True, help="对 IPG 生成的人像做水平翻转增强（将生成集翻倍）")
    parser.add_argument("--ipg_root", type=str, default="Pose2ID-main/IPG", help="Path to IPG root directory.")
    parser.add_argument("--ipg_ckpt_dir", type=str, default=None, help="Path to IPG pretrained directory (default: <ipg_root>/pretrained)")
    parser.add_argument("--ipg_pose_dir", type=str, default=None, help="Path to IPG pose directory (default: <ipg_root>/standard_poses)")
    parser.add_argument("--ipg_python", type=str, default=None, help="Path to IPG venv python (default: <ipg_root>/.venv-ipg/bin/python)")
    parser.add_argument("--ipg_num_poses", type=int, default=6, help="Number of generated poses to use for fusion")
    parser.add_argument("--ipg_eta", type=float, default=1.0, help="Fusion strength for IPG features")
    # 可选增强
    parser.add_argument("--aug_vflip", action='store_true', help="启用竖直翻转增强（人像默认不建议，特定场景再启用）。")
    parser.add_argument("--aug_rotate_deg", type=float, default=0.0, help="旋转增强角度（使用 ±deg；建议 5~15；0 关闭）。")
    
    args = parser.parse_args()
    main(args) 