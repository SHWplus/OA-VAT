# OA-VAT

[![arXiv](https://img.shields.io/badge/arXiv-2604.21453-b31b1b.svg)](https://arxiv.org/abs/2604.21453)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.2.2](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📢 News & Updates

- 🎉 **OA-VAT is accepted by CVPR 2026 as a Poster presentation!**

## 🛠 Prerequisites

Before installation, ensure you have the following:
- **Operating System:** Ubuntu 20.04/22.04/24.04 (Recommended) or Windows 10/11.
- **Hardware:** NVIDIA GPU (Recommended for real-time inference) with Drivers supporting CUDA 12.1+.
- **Package Manager:** [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

---

## 🚀 Installation & Environment Setup

Setting up the environment correctly is crucial due to specific dependencies on **PyTorch 2.2.2** and **CUDA 12.1**.

### 1. Clone the Repository
```bash
git clone https://github.com/SHWplus/OA-VAT.git
cd OA-VAT
```

### 2. Create Conda Environment
We provide a streamlined `environment.yml` that includes essential system libraries and Python dependencies.

```bash
# Create the environment from the provided yaml file
conda env create -f environment.yml

# Activate the environment
conda activate follow
```

### 3. Verify Deep Learning Backend (Important)
Since PyTorch with CUDA support often requires a specific index URL, run the following to ensure your GPU is recognized:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

*Note: If `CUDA available` returns `False`, manually reinstall the correct Torch version:*
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 🧩 Optional: IPG Environment (Pose2ID)

During the main pipeline, we additionally use **IPG** (Identity-Guided Pedestrian Generation) from the third-party repository **Pose2ID** (`Pose2ID/IPG`).

Since IPG has its own dependency stack (notably `torch==2.0.1` / `xformers==0.0.22` / `diffusers`), it is recommended to install it in a **separate conda environment**.

### 1. Create a Dedicated Conda Environment
```bash
conda create -n ipg python=3.9
conda activate ipg
```

### 2. Install IPG Python Dependencies
If you already have Pose2ID as a subfolder in this repo:

```bash
pip install -r Pose2ID/IPG/requirements.txt
```

If you prefer installing Pose2ID separately (recommended when you want to keep this repo lightweight), follow the official installation instructions:

- **Pose2ID**: `Pose2ID/README.md` (upstream)

### 3. Download IPG Pretrained Weights
IPG requires pretrained checkpoints. Please follow the upstream instructions to download:

- `sd-vae-ft-mse`
- `stable-diffusion-v1-5`
- Pose2ID IPG weights (place them under an IPG checkpoint directory, e.g. `Pose2ID/IPG/pretrained/`)

### 4. Quick Sanity Check
```bash
python Pose2ID/IPG/inference.py --ckpt_dir pretrained --pose_dir standard_poses --ref_dir ref --out_dir output
```

---

## 🧪 Optional: Simulation Environment (gym-unrealcv)

To evaluate the full pipeline in Unreal Engine simulation, you need to install **gym-unrealcv** (included in this repo under `gym-unrealcv/`).

### 1. Install Python Package
```bash
pip install -e ./gym-unrealcv
```

This will install the minimal dependencies required by `gym-unrealcv` (see `gym-unrealcv/setup.py`), including `gym==0.10.9`, `unrealcv`, and `opencv-python`.

### 2. Prepare Unreal Binary (Environment)
Download the Unreal environment binary via the provided script:

```bash
python gym-unrealcv/load_env.py -e {ENV_NAME}
```

For example, `ENV_NAME` can be `UrbanCity`, `Garden`, etc. The binary will be placed under `gym-unrealcv/gym_unrealcv/envs/UnrealEnv/`.

### 3. Minimal Example: Run Evaluation in UnrealCV
```bash
export PYTHONPATH=./ORTrack:$PYTHONPATH
python eval_in_unrealcv.py \
  --path_to_video "unrealcv:UnrealTrackMulti-UrbanCityNav-ContinuousColor-v0" \
  --episodes 20 \
  --desired_width 160 --desired_height 120 \
  --text_query "person" \
  --reference_feature_path /path/to/feature_set_dinov3.pt \
  --yolo_model_path /path/to/yoloe-11l-seg.pt \
  --yolo_confidence_thresh 0.1 \
  --ortrack_lost_thresh 0.5 \
  --reacq_match_thresh 0.5 \
  --match_similarity_thresh 0.4 \
  --online_enhance --kf_mode confaware \
  --unreal_seed 101 \
  --save_images_to ./outputs_unrealcv
```

---

## 🧠 Optional: Diffusion Policy (Training & Usage)

This project optionally supports a **Diffusion Policy** model for action planning (used by `main.py` when `--dp_enable` is set).
The implementation is based on the third-party repo included at `gym-unrealcv/diffusion_policy-main/`.

### 1. Install Diffusion Policy Environment (Recommended: Separate Conda Env)
Diffusion Policy has its own dependency stack. We recommend creating a dedicated environment.

```bash
conda env create -f gym-unrealcv/diffusion_policy-main/conda_environment.yaml
conda activate robodiff
```

For details, refer to the upstream README in this repo:

- `gym-unrealcv/diffusion_policy-main/README.md`

### 2. Collect Demonstrations in UnrealCV (gym-unrealcv)
Use `gym-unrealcv/tools/collect.py` to collect rollouts from UnrealCV tasks.
This script writes episode folders containing images and a `labels.jsonl` file.

Minimal example:

```bash
python gym-unrealcv/tools/collect.py \
  --env "UnrealTrackMulti-UrbanCityNav-ContinuousColor-v0" \
  --out_dir ./data/unreal_collect \
  --episodes 10
```

### 3. Convert Collected Data to ReplayBuffer (Zarr)
Convert the collected dataset into a Diffusion Policy compatible **ReplayBuffer** format:

```bash
python gym-unrealcv/tools/convert_unreal_to_replay.py \
  --input-root ./data/unreal_collect \
  --output-zarr ./data/unreal_replay.zarr \
  --image-size 96 96 \
  --horizon 8
```

### 4. Train / Evaluate Diffusion Policy
Training entrypoint is inside `gym-unrealcv/diffusion_policy-main/`.

```bash
cd gym-unrealcv/diffusion_policy-main
python train.py --config-dir=. --config-name=<your_config.yaml>
```

If you already have a checkpoint, you can evaluate it with:

```bash
python eval.py --checkpoint /path/to/model.ckpt --output_dir ./eval_outputs --device cuda:0
```

---

## 📦 Key Dependencies

This project relies on the following core libraries:
- **Inference:** `ultralytics` (YOLOv8), `diffusers`, `transformers`
- **Robotics/Sim:** `djitellopy`, `gym`, `gym-unrealcv`
- **Vision:** `opencv-python`, `scikit-image`
- **Math/Data:** `numpy`, `pandas`, `scipy`
- **UI/Control:** `pygame`, `pynput`

---

## 💻 Usage

### Quick Demo
Before running `main.py` or UnrealCV evaluation, you need to prepare a **reference feature file** (`feature_set_dinov3.pt`) for the target category using `extract_target_feature_dinov3.py`. This file will be passed via `--reference_feature_path`.

#### 0. Extract Target Reference Features (Required)

##### Case A: Non-person target (no Pose2ID/IPG used)
```bash
python extract_target_feature_dinov3.py \
  --image_folder /path/to/image_folder \
  --output_path /path/to/feature_set_dinov3.pt \
  --checkpoint_path weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --model_name dinov3_vitb16 \
  --classes <your_class_name> \
  --confidence 0.3 \
  --show_mask
```

##### Case B: Person target (use Pose2ID/IPG to improve identity cues)
This mode requires the optional **IPG environment** (see the section above) and a dedicated IPG Python interpreter.

```bash
export IPG_PY="/path/to/conda/envs/ipg/bin/python"
python extract_target_feature_dinov3.py \
  --image_folder /path/to/image_folder \
  --output_path /path/to/feature_set_dinov3.pt \
  --use_ipg_for_person \
  --ipg_root Pose2ID/IPG \
  --ipg_ckpt_dir Pose2ID/IPG/pretrained/pretrained \
  --ipg_pose_dir Pose2ID/IPG/standard_poses \
  --ipg_python ${IPG_PY} \
  --classes person \
  --confidence 0.10 \
  --show_mask \
  --model_name dinov3_vitb16 \
  --checkpoint_path weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

To run the basic object following script:
```bash
export PYTHONPATH=./ORTrack:$PYTHONPATH
python main.py \
  --path_to_video /path/to/video.mp4 \
  --desired_width 320 --desired_height 240 \
  --yolo_model_path "ORTrack/weights/yoloe-11l-seg.pt" \
  --text_query "person" \
  --yolo_confidence_thresh 0.2 \
  --ortrack_lost_thresh 0.5 \
  --reacq_match_thresh 0.5 \
  --kf_max_predict_frames 8 \
  --reference_feature_path /path/to/feature_set_dinov3.pt \
  --match_similarity_thresh 0.4 \
  --online_enhance --kf_mode confaware \
  --save_images_to ./outputs
```

### Training
If you wish to retrain the model on your custom dataset:
```bash
# Update the data.yaml path before running
python train.py --epochs 100 --batch 16
```

---

## 📂 Project Structure
```text
follow/
├── assets/             # Images and demo videos
├── configs/            # Configuration files (.yaml)
├── models/             # Pre-trained weights (.pt, .pth)
├── scripts/            # Helper scripts
├── environment.yml     # Conda environment definition
├── main.py             # Main entry point
└── README.md
```

---

## ❓ Troubleshooting

**1. OpenCV/Qt Errors:**
If you encounter `libGL.so.1` or Qt plugin errors on Linux, install the following system packages:
```bash
sudo apt-get update && sudo apt-get install libgl1-mesa-glx libqt5gui5 -y
```

**2. ModuleNotFoundError:**
If a specific library is missing, you can install the complete list from the backup requirements:
```bash
pip install -r requirements.txt
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📚 Citation

If you find OA-VAT useful in your research or applications, please cite it using the following BibTeX:

```bibtex
  @misc{sun2026instancelevelvisualactivetracking,
      title={Instance-level Visual Active Tracking with Occlusion-Aware Planning}, 
      author={Haowei Sun and Kai Zhou and Hao Gao and Shiteng Zhang and Jinwu Hu and Xutao Wen and Qixiang Ye and Mingkui Tan},
      year={2026},
      eprint={2604.21453},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.21453}, 
}
```