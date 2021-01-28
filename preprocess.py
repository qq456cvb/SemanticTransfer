import xml.etree.ElementTree as ET
import os
import OpenEXR
from pathlib import Path
import numpy as np
import cv2
import math
from tqdm import tqdm
import pymesh


def find_env_maps(rootdir):
    pathlist = Path(rootdir).glob('*/*.xml')
    env_maps = set()
    for path in pathlist:
        doc = ET.parse(str(path))
        root = doc.getroot()
        emitter = root.find('emitter')
        elem = emitter.find('string')
        if elem is not None:
            fn = elem.attrib['value']
            if fn.find('envmap') >= 0:
                env_maps.add(fn.split('/')[-1])
    env_maps = list(env_maps)
    env_maps.sort()
    with open('hdrs.txt', 'w') as f:
        f.writelines('\n'.join(env_maps))


def gen_azimuth_elevation(rootdir):
    pathlist = Path(rootdir).glob('*/*.xml')
    for path in tqdm(pathlist):
        doc = ET.parse(str(path))
        root = doc.getroot()
        sensor = root.find('sensor')
        elem = sensor.find('transform').find('lookAt')
        assert(elem is not None)
        if elem is not None:
            origin = elem.attrib['origin']
            origin = [float(v) for v in origin.split(',')]
            elevation = np.arccos(origin[1] / 2.2)
            azimuth = np.arctan2(origin[0], origin[2])
            if azimuth < 0:
                azimuth += 2 * np.pi
            np.save(os.path.join('/home/neil/disk/shapenet20views', str(path).split('genre-xml_v2/')[-1].replace('.xml', '_azimuth.npy')), azimuth)
            np.save(os.path.join('/home/neil/disk/shapenet20views', str(path).split('genre-xml_v2/')[-1].replace('.xml', '_elevation.npy')), elevation)
            # print(os.path.join('/data/shapenet20views', str(path).split('genre-xml_v2/')[-1].replace('.xml', '_azimuth.npy')))
            # exit()

def clear(rootdir):
    pathlist = Path(rootdir).glob('*/*_voxel2renderer.npy')
    for path in tqdm(pathlist):
        if os.path.exists(path):
            print("removed")
            os.remove(path)
        else:
            print("The file does not exist")
            
def gen_transform_matrix(rootdir):
    pathlist = Path(rootdir).glob('*/*_voxel_normalized_128.mat')
    for path in tqdm(pathlist):
        cls_name = str(path).split('/')[-3]
        model_name = str(path).split('/')[-2]
        mesh = pymesh.load_mesh('/home/neil/disk/ShapeNetCore.v2/{}/{}/models/model_normalized.obj'.format(cls_name, model_name))
        vertices = np.array(mesh.vertices)
        
        v2 = vertices.copy()
        v2[:, 2] = vertices[:, 1]
        v2[:, 1] = -vertices[:, 2]

        xmax, xmin = v2[:, 0].max(), v2[:, 0].min()
        ymax, ymin = v2[:, 1].max(), v2[:, 1].min()
        zmax, zmin = v2[:, 2].max(), v2[:, 2].min()

        center = np.array([(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2])
        scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
        
        shape2voxel = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]) \
            @ np.array([[1. / scale, 0, 0, 0], [0, 1. / scale, 0, 0], [0, 0, 1. / scale, 0], [0, 0, 0, 1]]) \
            @ np.array([[1, 0, 0, -center[0]], [0, 1, 0, -center[1]], [0, 0, 1, -center[2]], [0, 0, 0, 1]]) \
            @ np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        renderer2shape = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        renderer2voxel = shape2voxel @ renderer2shape
        
        np.save(str(path).replace('_voxel_normalized_128.mat', '_renderer2voxel.npy'), renderer2voxel)
        np.save(str(path).replace('_voxel_normalized_128.mat', '_voxel2renderer.npy'), np.linalg.inv(renderer2voxel))


def readexr(fn):
    img = OpenEXR.InputFile(fn)
    # print(img.header())
    return img.channels(["color.R", "color.G", "color.B"]), \
        img.channels(["normal.R", "normal.G", "normal.B"]), \
        img.channels(["depth.R", "depth.G", "depth.B"])

def bytes2np(bytes, width=480, height=480):
    return np.stack([np.frombuffer(bytes[i], dtype=np.float16) for i in range(3)], axis=-1).reshape(height, width, 3).astype(np.float32)


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

if __name__ == "__main__":
    # root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '03001627')
    # clear('/home/neil/disk/shapenet20views/03001627')
    # gen_transform_matrix('/home/neil/disk/shapenet20views/03001627')
    # for _ in range(20000):
    #     real = np.random.uniform() * 2 * np.pi
    #     prob = real2prob(real, 2 * np.pi, 24, True)
    #     diff = np.abs(prob2real(prob, 2 * np.pi, 24, True) - real)
    #     assert(diff < 1e-3)

    # for _ in range(20):
    #     real = np.random.uniform() * np.pi
    #     prob = real2prob(real, np.pi, 12, False)
    #     diff = np.abs(prob2real(prob, np.pi, 12, False) - real)
    #     print(prob2real(prob, np.pi, 12, False), real)
    #     assert(diff < 1e-3)
    # find_env_maps(os.path.join(os.path.dirname(os.path.abspath(__file__)), '03001627'))
    gen_azimuth_elevation(os.path.join(os.path.dirname(os.path.abspath(__file__)), '03001627'))
    # rgb, normal, depth = readexr(os.path.join(os.path.dirname(os.path.abspath(__file__)), '03001627', '1a6f615e8b1b5ae4dbbc9440457e303e/03001627_1a6f615e8b1b5ae4dbbc9440457e303e_view004.exr'))


    