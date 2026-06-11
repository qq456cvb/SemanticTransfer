# SemanticTransfer
Code repo for the paper [Semantic Correspondence via 2D-3D-2D Cycle](https://arxiv.org/abs/2004.09061).

<!-- README refined by Cursor -->

## Data and Artifact Mirrors

No verified Hugging Face mirror is available yet for the artifacts below; use the original sources until a complete mirror is uploaded.

Original, external, or pending sources:
- Pretrained weights: [https://drive.google.com/drive/folders/1VN4dIrMqtIxb0CJleOx7aco21BUSL9qp?usp=sharing](https://drive.google.com/drive/folders/1VN4dIrMqtIxb0CJleOx7aco21BUSL9qp?usp=sharing). Hugging Face mirror is pending because the source did not expose a retrievable public file URL during this cleanup.

# Demo
Please run `demo.py`.

# Pretrained Weights
You can download them from [Google Drive](https://drive.google.com/drive/folders/1VN4dIrMqtIxb0CJleOx7aco21BUSL9qp?usp=sharing).

# Training

Training the full pipeline is somewhat involved and complicated, and our code is heavily based on [ShapeHD](https://github.com/xiumingzhang/GenRe-ShapeHD). In general, there are four steps:

- Train ShapeHD model as outlined [here](https://github.com/xiumingzhang/GenRe-ShapeHD#shapehd-1).
- Prepare synthetic ShapeNet model renderings by ``mitsuba`` and generate their corresponding viewpoints through ``preprocess.py``.
- Train viewpoint estimation network by running ``scripts/train_vp.sh``.
- Train 3D embedding prediction network by running ``train_embs.py`` and then generate keypoints' average embeddings for retrieval. This step requires [KeypointNet](https://github.com/qq456cvb/KeypointNet) dataset.
