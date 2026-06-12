# Semantic Correspondence via 2D-3D-2D Cycle

[![arXiv](https://img.shields.io/badge/arXiv-2004.09061-b31b1b.svg)](https://arxiv.org/abs/2004.09061)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/qq456cvb/SemanticTransfer)

Official implementation of [Semantic Correspondence via 2D-3D-2D Cycle](https://arxiv.org/abs/2004.09061).

Instead of training correspondences directly in 2D, this method lifts the problem to 3D: a single-view image is reconstructed into a 3D shape (via 2.5D sketches), its viewpoint is estimated, dense 3D semantic embeddings are predicted, and keypoint labels are transferred from the [KeypointNet](https://github.com/qq456cvb/KeypointNet) dataset through 3D retrieval before being projected back into the image. Reasoning in 3D lets the model handle self-occlusion and visibility explicitly.

## Pipeline

1. **2.5D sketch estimation** (`models/marrnet1.py`) — depth, normals, and silhouette from a masked RGB image
2. **3D shape completion** (`models/shapehd.py`) — voxel shape from the 2.5D sketches ([ShapeHD](https://github.com/xiumingzhang/GenRe-ShapeHD))
3. **Viewpoint estimation** (`models/viewpoint.py`) — azimuth/elevation of the input view
4. **Dense 3D embeddings** (`models/dense_embedding.py`) — per-point semantic embeddings matched against KeypointNet keypoint embeddings (`data/embeddings_kpnet_norm.pkl`), then rendered back to 2D

## Pretrained Weights

Download the checkpoints from [Hugging Face](https://huggingface.co/qq456cvb/SemanticTransfer) into the `weights/` folder:

```bash
hf download qq456cvb/SemanticTransfer marrnet1.pt shapehd.pt best.pt --local-dir weights
```

(`weights/embeddings_norm.pt` is already included in the repository.)

Google Drive mirror: [link](https://drive.google.com/drive/folders/1VN4dIrMqtIxb0CJleOx7aco21BUSL9qp?usp=sharing).

## Demo

```bash
python demo.py
```

Runs the full pipeline on the bundled example (`data/demo_rgb.png` + `data/demo_mask.png`) and visualizes the transferred keypoints. Requires PyTorch, [neural_renderer](https://github.com/daniilidis-group/neural_renderer), hydra, scikit-image, and OpenCV.

## Training

Training the full pipeline is somewhat involved, and our code is heavily based on [ShapeHD](https://github.com/xiumingzhang/GenRe-ShapeHD). In general, there are four steps:

1. Train the ShapeHD model as outlined [here](https://github.com/xiumingzhang/GenRe-ShapeHD#shapehd-1).
2. Prepare synthetic ShapeNet model renderings with `mitsuba` and generate their corresponding viewpoints through `preprocess.py`.
3. Train the viewpoint estimation network with `scripts/train_vp.sh`.
4. Train the 3D embedding prediction network with `train_emb.py`, then generate the keypoints' average embeddings for retrieval. This step requires the [KeypointNet](https://github.com/qq456cvb/KeypointNet) dataset.

## Citation

```bibtex
@article{you2020semantic,
  title={Semantic Correspondence via 2D-3D-2D Cycle},
  author={You, Yang and Li, Chengkun and Lou, Yujing and Cheng, Zhoujun and Ma, Lizhuang and Lu, Cewu and Wang, Weiming},
  journal={arXiv preprint arXiv:2004.09061},
  year={2020}
}
```
