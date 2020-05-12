from glob import glob
import numpy as np
import torch.utils.data as data
import cv2
import os
import hydra.utils


def imread_wrapper(*args, output_channel_order='RGB'):
    """
    Convinience wrapper for cv2.imread() that can return result in RGB order

    Args:
        *args: Positional parameters that imread() takes
            See documentation for cv2.imread()
        output_channel_order: Whether to output RGB or BGR orders; has effects only when
            number of channels is three or four (fourth being alpha)
            'RGB' or 'BGR' (case-insensitive)
            Optional; defaults to 'RGB'

    Returns:
        im: Loaded image
            Numpy array of shape (m, n) or (m, n, c)
    """
    output_channel_order = output_channel_order.lower()
    assert ((output_channel_order == 'rgb') or (output_channel_order == 'bgr')), \
        "'output_channel_order' has to be either 'RGB' or 'BGR' (case-insensitive)"

    im = cv2.imread(*args)
    assert (im is not None), "%s not existent" % args[0]

    if (im.ndim == 3) and (output_channel_order == 'rgb'):
        if im.shape[2] == 3:
            im = im[:, :, ::-1]
        elif im.shape[2] == 4:  # with alpha
            im = im[:, :, [2, 1, 0, 3]]
    return im


class Dataset(data.Dataset):
    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    def __init__(self, opt, model):
        # Get required keys and preprocessing from the model
        required = model.requires
        self.preproc = model.preprocess_wrapper
        # Wrapper usually crops and resizes the input image (so that it's just
        # like our renders) before sending it to the actual preprocessing

        # Associate each data type required by the model with input paths
        type2filename = {}
        for k in required:
            type2filename[k] = getattr(opt, 'input_' + k)

        # Generate a sorted filelist for each data type
        type2files = {}
        for k, v in type2filename.items():
            type2files[k] = sorted(glob(os.path.join(hydra.utils.get_original_cwd(), v)))
        ns = [len(x) for x in type2files.values()]
        assert len(set(ns)) == 1, \
            ("Filelists for different types must be of the same length "
             "(1-to-1 correspondance)")
        self.length = ns[0]

        samples = []
        for i in range(self.length):
            sample = {}
            for k, v in type2files.items():
                sample[k + '_path'] = v[i]
            samples.append(sample)
        self.samples = samples
        self.type2files = type2files

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        sample = self.samples[i]

        # print(sample)
        # Actually loading the item
        sample_loaded = {}
        for k, v in sample.items():
            sample_loaded[k] = v  # as-is
            if k == 'rgb_path':
                im = imread_wrapper(
                    v, cv2.IMREAD_COLOR, output_channel_order='RGB')
                # Normalize to [0, 1] floats
                im = im.astype(float) / float(np.iinfo(im.dtype).max)
                sample_loaded['rgb'] = im
            elif k == 'mask_path':
                im = imread_wrapper(
                    v, cv2.IMREAD_GRAYSCALE)
                # Normalize to [0, 1] floats
                im = im.astype(float) / float(np.iinfo(im.dtype).max)
                sample_loaded['silhou'] = im
            else:
                raise NotImplementedError(v)

        # Preprocessing specified by the model
        sample_loaded = self.preproc(sample_loaded)
        sample_loaded['rgb_fn'] = self.type2files['rgb'][i]
        sample_loaded['mask_fn'] = self.type2files['mask'][i]
        # Convert all types to float32 for faster copying
        self.convert_to_float32(sample_loaded)
        return sample_loaded

    @staticmethod
    def convert_to_float32(sample_loaded):
        for k, v in sample_loaded.items():
            if isinstance(v, np.ndarray):
                if v.dtype != np.float32:
                    sample_loaded[k] = v.astype(np.float32)
