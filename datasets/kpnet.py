import torch
import os
import json
from glob import glob
import numpy as np
import hydra


BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')
ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel",}

NAMES2ID = {v: k for k, v in ID2NAMES.items()}


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise


def normalize_pc(pc):
    bounds = np.stack([np.min(pc, 0), np.max(pc, 0)], 0)
    center, scale = (bounds[1] + bounds[0]) / 2, np.max(bounds[1] - bounds[0])
    pc = (pc - center) / scale
    return pc, center, scale


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.catg = cfg.class_name
        self.cfg = cfg
            
        annots = json.load(open(hydra.utils.to_absolute_path(cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[cfg.class_name]]
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        
        self.pcds = []
        self.keypoints = []
        self.mesh_names = []
        for fn in glob(os.path.join(hydra.utils.to_absolute_path(cfg.data.pcd_root), NAMES2ID[cfg.class_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            
            curr_keypoints = -np.ones((self.nclasses,), dtype=np.int)
            for i, kp in enumerate(keypoints[model_id]):
                curr_keypoints[kp[1]] = kp[0]
            self.keypoints.append(curr_keypoints)
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        label = self.keypoints[idx]
        
        pc, center, scale = normalize_pc(pc)
            
        return pc.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.pcds)