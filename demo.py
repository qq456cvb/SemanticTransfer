import os
import time
from shutil import rmtree
from tqdm import tqdm
import torch
from datasets import Dataset
from scipy.io import savemat
import numpy as np
from models import viewpoint, shapehd, dense_embedding
import cv2
from skimage import measure
import neural_renderer as nr
import loggers
from utils import sample_vertex_from_mesh
import json
import seaborn as sns
import pickle
# import pymesh
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import imageio
import hydra
import hydra.utils
import math
    
    

def real2prob(val, max_val, num_bins, circular=False):
    res = np.zeros((num_bins,))
    if val >= max_val:
        val -= 1e-7
        assert val < max_val
    if not circular:
        interval = max_val / (num_bins - 1)
        low = math.floor(val / interval)
        high = low + 1
        assert low >= 0 and high < num_bins
        res[low] = 1. - (val / interval - low)
        res[high] = 1. - res[low]
        assert 0 <= res[low] <= 1.
        return res
    else:
        interval = max_val / num_bins
        if val < interval / 2:
            val += max_val
        res = real2prob(val - interval / 2, max_val, num_bins + 1)
        res[0] += res[-1]
        return res[:-1]

def prob2real(prob, max_val, num_bins, circular=False):
    if not circular:
        return np.sum(prob * np.arange(num_bins) * max_val / (num_bins - 1))
    else:
        interval = max_val / num_bins
        vecs = np.stack([np.cos(np.arange(num_bins) * interval + interval / 2), np.sin(np.arange(num_bins) * interval + interval / 2)], axis=-1)
        res = np.sum(np.expand_dims(prob, axis=-1) * vecs, axis=0)
        res = np.arctan2(res[1], res[0])
        if res < 0:
            res += 2 * np.pi  # remap to [0, 2pi]
        return res
    

def vol2obj(df, th=0.25):
    if th < np.min(df):
        df[0, 0, 0] = th - 1
    if th > np.max(df):
        df[-1, -1, -1] = th + 1
    spacing = (1 / 128, 1 / 128, 1 / 128)
    verts, faces, _, _ = measure.marching_cubes_lewiner(
        df, th, spacing=spacing)
    verts -= np.array([0.5, 0.5, 0.5])
    return verts, faces
    
def project(data, x, y, z):
    p_3d = np.array([[x], [y], [z], [1.]])
    RT = np.append(data['rot_mat'], np.resize(data['trans_mat'], [3, 1]), axis=1)
    # Here we convert focal length in mm into focal length in pixels.
    f_pix = data['focal_length'] / 32. * data['img_size'][0]
    K = np.array([[f_pix, 0., data['img_size'][0] / 2.], [0, f_pix, data['img_size'][1] / 2.], [0., 0., 1.]])
    p_2d = np.dot(np.dot(K, RT), p_3d)
    p_2d = p_2d / p_2d[2]
    # Convert u, v into conventional image coordinates (from +u: leftward, +v: upward to +u: rightward, +v: downward).
    p_2d = data['img_size'] - p_2d[:2, 0]
    return p_2d


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))
    
@hydra.main(config_path='config/config.yaml', strict=False)
def main(opt):
    work_dir = hydra.utils.get_original_cwd()
    logger_list = [
        loggers.TerminateOnNaN(),
    ]
    logger = loggers.ComposeLogger(logger_list)
    
    model = shapehd.Model_test(opt, logger)
    model.cuda()
    model.eval()
    
    net_aziele = viewpoint.Model(opt, logger)
    net_aziele.load_state_dict(os.path.join(work_dir, 'weights/best.pt'))
    
    net_aziele.cuda()
    net_aziele.eval()
    
    # pointnet
    predictor = dense_embedding.Model().cuda()
    predictor.load_state_dict(torch.load(os.path.join(work_dir, 'weights/embeddings_norm.pt'))['state_dict'])
    
    predictor.eval()
    
    dataset = Dataset(opt, model)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        shuffle=False
    )
    
    embeddings = pickle.load(open(os.path.join(work_dir, 'data/embeddings_kpnet_norm.pkl'), 'rb'))
    
    renderer = nr.Renderer(camera_mode='look_at', viewing_angle=45)
    for batch in dataloader:
        # Forward MarrNet-1
        pred1 = model.marrnet1.predict(batch, load_gt=False, no_grad=True)
        
        pred_normal = pred1['normal'][0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1]
        pred_depth = pred1['depth'][0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1]
        pred_silhou = model.postprocess(pred1['silhou']).cpu().numpy()[0].transpose((1, 2, 0))[:, :, ::-1]
        pred_silhou[pred_silhou >= model.pred_silhou_thres] = 1
        pred_silhou[pred_silhou < model.pred_silhou_thres] = 0
        
        gt_silhou = cv2.imread(batch['mask_fn'][0]).astype(np.float)
        gt_silhou = cv2.resize(gt_silhou, (256, 256))
        gt_silhou /= 255.
        
        pred_normal = MinMaxScaler().fit_transform(pred_normal.reshape(-1, 3)).reshape(256, 256, 3) * 255.
        pred_depth = MinMaxScaler().fit_transform(pred_depth.reshape(-1, 1)).reshape(256, 256, 1) * 255.
        
        pred = net_aziele.predict(batch, load_gt=False, no_grad=True)
        azimuth = prob2real(softmax(pred['azimuth'][0].cpu().numpy()), 2 * np.pi, 24, True)
        elevation = prob2real(softmax(pred['elevation'][0].cpu().numpy()), np.pi, 12, False)
        
        # Forward MarrNet-2
        for net_name in ('marrnet2', 'marrnet2_noft'):
            net = getattr(model.net, net_name)
            net.silhou_thres = model.pred_silhou_thres * model.scale_25d

        model.input_names = ['depth', 'normal', 'silhou']
        pred2 = model.predict(pred1, load_gt=False, no_grad=True)
        voxel = pred2['voxel']
        voxel = 1 / (1 + torch.exp(-voxel))[0, 0]
        voxel = voxel.cpu().numpy()
        voxel = np.transpose(voxel, (0, 2, 1))
        voxel = np.flip(voxel, 2)
        
        verts, faces = vol2obj(voxel, th=0.1)  # fine tune threshold?
        
        def to_obj_str(verts, faces):
            text = ""
            for p in verts:
                text += "v "
                for x in p:
                    text += "{} ".format(x)
                text += "\n"
            for f in faces:
                text += "f "
                for x in f:
                    text += "{} ".format(x + 1)
                text += "\n"
            return text
        
        obj_str = to_obj_str(verts, faces)
        with open('output.obj', 'w') as f:
            f.write(obj_str)
        
        pcd, _, _, _ = sample_vertex_from_mesh(verts, faces, num_samples=2048)  # in shapenet coordinates
        
        verts = torch.from_numpy(verts).float().cuda()
        faces = torch.from_numpy(faces.copy()).cuda()
        verts = verts[None]
        faces = faces[None]
        
        textures = torch.ones(1, faces.shape[1], 2, 2, 2, 3, dtype=torch.float32).cuda()
        
        dist = opt.init_dist
        
        imgsize = 480
        K = np.array([[imgsize / 2, 0, imgsize / 2 - 0.5],
                        [0, imgsize / 2, imgsize / 2 - 0.5],
                        [0, 0, 1]])

        eye_y = dist * np.cos(elevation)
        eye_x = dist * np.sin(elevation) * np.sin(azimuth)
        eye_z = dist * np.sin(elevation) * np.cos(azimuth)
            
        R = np.zeros((3, 3))
        R[:, 1] = np.array([0, 1, 0])
        R[:, 2] = -np.array([eye_x, eye_y, eye_z])
        R[:, 2] /= np.linalg.norm(R[:, 2])
        R[:, 0] = np.cross(R[:, 1], R[:, 2])
        R[:, 0] /= np.linalg.norm(R[:, 0])
        R[:, 1] = np.cross(R[:, 2], R[:, 0])
        R[:, 1] /= np.linalg.norm(R[:, 1])  # left hand coord
        
        scale = torch.tensor(1., dtype=torch.float, requires_grad=True, device='cuda')
        renderer.eye = torch.tensor([eye_x, eye_y, eye_z], dtype=torch.float, requires_grad=False, device='cuda') * scale
        
        if opt.post_opt:
            print('fine-tuning view points...')
            optimizer = torch.optim.Adam([scale], lr=0.01)
        
            pred_silhou = model.postprocess(pred1['silhou'].detach().cuda())
            pred_silhou = torch.clamp(pred_silhou, 0, 1)  # 1 x 1 x 256 x 256
            
            pred_silhou[pred_silhou > model.pred_silhou_thres] = 1
            pred_silhou[pred_silhou <= model.pred_silhou_thres] = 0
            writer = imageio.get_writer(os.path.join('opt.gif'), mode='I')
            for k in tqdm(range(30)):
                optimizer.zero_grad()
                renderer.eye = torch.tensor([eye_x, eye_y, eye_z], dtype=torch.float, requires_grad=False, device='cuda') * scale
                image = renderer(verts, faces, textures, mode='silhouettes')  # 1 x 256 x 256
                
                image = torch.flip(image, [2])
                
                loss = torch.mean((image - pred_silhou[0]) ** 2)
                loss.backward()
                optimizer.step()
                
                writer.append_data((255 * image.detach().cpu().numpy()[0]).astype(np.uint8))
            writer.close()
            
            eye_x = renderer.eye[0].item()
            eye_y = renderer.eye[1].item()
            eye_z = renderer.eye[2].item()

        extrinsic = np.concatenate([R.T, -R.T @ np.array([eye_x, eye_y, eye_z])[:, None]], axis=1)
        
        # pass through embedding network
        with torch.no_grad():
            pcd_embeddings = predictor(torch.from_numpy(pcd[None]).float())[0].cpu().numpy()  # pretrained model is unnormalized
            
        target_embeddings = np.array([embeddings[i] for i in range(21)])
        
        dists = np.linalg.norm(target_embeddings[:, None, :] - pcd_embeddings[None], axis=-1)
        kps_pred = pcd[np.argmin(dists, axis=1)]
        
        kps_projection = (K @ extrinsic @ np.concatenate([kps_pred, np.ones([kps_pred.shape[0], 1])], axis=1).T).T
        kps_projection[:, 0] = kps_projection[:, 0] / kps_projection[:, 2]
        kps_projection[:, 1] = kps_projection[:, 1] / kps_projection[:, 2]
        kps_projection[:, 0] = imgsize - 1 - kps_projection[:, 0]
        kps_projection[:, 1] = imgsize - 1 - kps_projection[:, 1]
        
        
        ################################################### visualize
        image, _, _ = renderer(verts, faces, textures)
        image = image.detach().cpu().numpy()[0].transpose((1, 2, 0))
        image = cv2.resize(image, (imgsize, imgsize))
        image = np.flip(image, 1)
        rgb = batch['rgb_crop'][0].cpu().numpy()
        mask = np.tile((image.sum(-1) == 0)[..., None], (1, 1, 3)).astype(np.float32)
        toshow = rgb * mask + image * (1 - mask)
        
        palette = sns.color_palette("hls", kps_pred.shape[0])
        for j, proj in enumerate(kps_projection):
            if 0 <= int(proj[1]) < imgsize and 0 <= int(proj[0]) < imgsize:
                # hack: demo png does not have chair arms
                if j in [6, 7, 8, 9, 14]:
                    continue
                color = tuple([c for c in palette[j]])
                cv2.circle(toshow, (int(proj[0]), int(proj[1])), 9, color=color, thickness=-1)
        cv2.imshow('result', toshow[:, :, ::-1])
        cv2.waitKey()
        ###################################################


if __name__ == "__main__":
    main()
    
    