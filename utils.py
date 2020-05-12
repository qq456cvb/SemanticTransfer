class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


str_stage = bcolors.OKBLUE + '==>' + bcolors.ENDC
str_verbose = bcolors.OKGREEN + '[Verbose]' + bcolors.ENDC
str_warning = bcolors.WARNING + '[Warning]' + bcolors.ENDC
str_error = bcolors.FAIL + '[Error]' + bcolors.ENDC


from copy import deepcopy
import numpy as np
import cv2


def imwrite_wrapper(*args, input_channel_order='RGB'):
    """
    Convinience wrapper for cv2.imwrite() that can write RGB image correctly

    Args:
        *args: Positional parameters that imwrite() takes
            See documentation for cv2.imwrite()
        input_channel_order: Whether the input is in RGB or BGR orders; has effects
            only when number of channels is three or four (fourth being alpha)
            'RGB' or 'BGR' (case-insensitive)
            Optional; defaults to 'RGB'
    """
    input_channel_order = input_channel_order.lower()
    assert ((input_channel_order == 'rgb') or (input_channel_order == 'bgr')), \
        "'input_channel_order' has to be either 'RGB' or 'BGR' (case-insensitive)"

    im = args[1]

    if (im.ndim == 3) and (input_channel_order == 'rgb'):
        if im.shape[2] == 3:
            im = im[:, :, ::-1]
        elif im.shape[2] == 4:  # with alpha
            im = im[:, :, [2, 1, 0, 3]]

    args_list = list(args)
    args_list[1] = im
    args_tuple = tuple(args_list)

    cv2.imwrite(*args_tuple)


def resize(im, target_size, which_dim, interpolation='bicubic', clamp=None):
    """
    Resize one dimension of the image to a certain size while maintaining the aspect ratio

    Args:
        im: Image to resize
            Any type that cv2.resize() accepts
        target_size: Target horizontal or vertical dimension
            Integer
        which_dim: Which dimension to match target_size
            'horizontal' or 'vertical'
        interpolation: Interpolation method
            'bicubic'
            Optional; defaults to 'bicubic'
        clamp: Clamp the resized image with minimum and maximum values
            Array_likes of one smaller float and another larger float
            Optional; defaults to None (no clamping)

    Returns:
        im_resized: Resized image
            Numpy array with new horizontal and vertical dimensions
    """
    h, w = im.shape[:2]

    if interpolation == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    else:
        raise NotImplementedError(interpolation)

    if which_dim == 'horizontal':
        scale_factor = target_size / w
    elif which_dim == 'vertical':
        scale_factor = target_size / h
    else:
        raise ValueError(which_dim)

    im_resized = cv2.resize(im, None, fx=scale_factor, fy=scale_factor,
                            interpolation=interpolation)

    if clamp is not None:
        min_val, max_val = clamp
        im_resized[im_resized < min_val] = min_val
        im_resized[im_resized > max_val] = max_val

    return im_resized


def alpha_blend(im1, im2, alpha):
    """
    Alpha blending of two images or one image and a scalar

    Args:
        im1, im2: Image or scalar
            Numpy array and a scalar or two numpy arrays of the same shape
        alpha: Weight of im1
            Float ranging usually from 0 to 1

    Returns:
        im_blend: Blended image -- alpha * im1 + (1 - alpha) * im2
            Numpy array of the same shape as input image
    """
    im_blend = alpha * im1 + (1 - alpha) * im2

    return im_blend


def rgb2gray(rgb):
    """
    Convert a RGB image to a grayscale image
        Differences from cv2.cvtColor():
            1. Input image can be float
            2. Output image has three repeated channels, other than a single channel

    Args:
        rgb: Image in RGB format
            Numpy array of shape (h, w, 3)

    Returns:
        gs: Grayscale image
            Numpy array of the same shape as input; the three channels are the same
    """
    ch = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gs = np.dstack((ch, ch, ch))

    return gs


def adjust_image_attribute(rgb, attr, d, random=False):
    """
    Adjust or randomize the specified attribute of the image

    Args:
        rgb: Image in RGB format
            Numpy array of shape (h, w, 3)
        attr: Image attribute to adjust or randomize
            'brightness', 'saturation', or 'contrast'
        d: If random, d must be positive, and alpha for blending is randomly drawn from
            [1 - d, 1 + d]; else, alpha will be just 1 + d
            Float
        random: Whether to set or randomize the attribute
            Boolean
            Optional; defaults to False

    Returns:
        rgb_out: Output image in RGB format
            Numpy array of the same shape as input
    """
    gs = rgb2gray(rgb)

    if random:
        assert (
            d > 0), "'d' must be positive for range [1 - d, 1 + d] to be valid"
        alpha = 1 + np.random.uniform(low=-d, high=d)
    else:
        alpha = 1 + d

    if attr == 'contrast':
        rgb_out = alpha_blend(rgb, np.mean(gs[:, :, 0]), alpha)
    elif attr == 'saturation':
        rgb_out = alpha_blend(rgb, gs, alpha)
    elif attr == 'brightness':
        rgb_out = alpha_blend(rgb, 0, alpha)
    else:
        raise NotImplementedError(attr)

    return rgb_out


def jitter_colors(rgb, d_brightness=0, d_contrast=0, d_saturation=0):
    """
    Color jittering by randomizing brightness, contrast and saturation, in random order

    Args:
        rgb: Image in RGB format
            Numpy array of shape (h, w, 3)
        d_brightness, d_contrast, d_saturation: Alpha for blending drawn from [1 - d, 1 + d]
            Nonnegative float
            Optional; defaults to 0, i.e., no randomization

    Returns:
        rgb_out: Color-jittered image in RGB format
            Numpy array of the same shape as input
    """
    attrs = ['brightness', 'contrast', 'saturation']
    ds = [d_brightness, d_contrast, d_saturation]

    # In random order
    ind = np.array(range(len(attrs)))
    np.random.shuffle(ind)  # in-place

    rgb_out = deepcopy(rgb)
    for idx in ind:
        rgb_out = adjust_image_attribute(
            rgb_out, attrs[idx], ds[idx], random=True)

    return rgb_out


def add_lighting_noise(rgb_0to1,
                       alpha_std,
                       eigvals=(0.2175, 0.0188, 0.0045),
                       eigvecs=((-0.5675, 0.7192, 0.4009),
                                (-0.5808, -0.0045, -0.8140),
                                (-0.5836, -0.6948, 0.4203))):
    """
    Add AlexNet-style PCA-based noise

    Args:
        rgb_0to1: Image in RGB format, normalized within [0, 1]; values can fall outside [0, 1] due to
            some preceding processing, but eigenvalues/vectors should match the magnitude order
            Numpy array of shape (h, w, 3)
        alpha_std: Standard deviation of the Gaussian from which alpha is drawn
            Positive float
        eigvals, eigvecs: Eigenvalues and their eigenvectors
            Array_likes of length 3 and shape (3, 3), respectively
            Optional; default to results from AlexNet

    Returns:
        rgb_0to1_out: Output image in RGB format, with lighting noise added
            Numpy array of the same shape as input
    """
    assert (rgb_0to1.dtype.name ==
            'float64'), "Input image must be normalized and hence be float"
    assert (alpha_std > 0), "Standard deviation must be positive"

    eigvals = np.array(eigvals)
    eigvecs = np.array(eigvecs)

    alpha = np.random.normal(loc=0, scale=alpha_std, size=3)
    noise_rgb = \
        np.sum(
            np.multiply(
                np.multiply(
                    eigvecs,
                    np.tile(alpha, (3, 1))
                ),
                np.tile(eigvals, (3, 1))
            ),
            axis=1
        )

    rgb_0to1_out = deepcopy(rgb_0to1)
    for i in range(3):
        rgb_0to1_out[:, :, i] += noise_rgb[i]

    return rgb_0to1_out


def normalize_colors(rgb_0to1, mean_rgb=(0.485, 0.456, 0.406), std_rgb=(0.229, 0.224, 0.225)):
    """
    Normalize colors

    Args:
        rgb_0to1: Image in RGB format, normalized within [0, 1]; values can fall outside [0, 1] due to
            some preceding processing, but mean and standard deviation should match the magnitude order
            Numpy array of shape (h, w, 3)
        mean_rgb, std_rgb: Mean and standard deviation for RGB channels
            Array_likes of length 3
            Optional; default to results computed from a random subset of ImageNet training images

    Returns:
        rgb_0to1_out: Output image in RGB format, with channels normalized
            Numpy array of the same shape as input
    """
    assert ('float' in rgb_0to1.dtype.name), "Input image must be normalized and hence be float"
    assert rgb_0to1.ndim == 3, "Nx3xHxW? This function was written for HxWx3"

    rgb_0to1_out = deepcopy(rgb_0to1)
    for i in range(3):
        rgb_0to1_out[:, :, i] = (
            rgb_0to1_out[:, :, i] - mean_rgb[i]) / std_rgb[i]

    return rgb_0to1_out


def denormalize_colors(rgb_norm, mean_rgb=(0.485, 0.456, 0.406), std_rgb=(0.229, 0.224, 0.225)):
    """
    Denormalize colors

    Args:
        rgb_norm: Image in RGB format, normalized by normalize_colors()
            Numpy array of shape (h, w, 3)
        mean_rgb, std_rgb: Mean and standard deviation for RGB channels used
            Array_likes of length 3
            Optional; default to results computed from a random subset of ImageNet training images

    Returns:
        rgb_0to1_out: Output image in RGB format, with channels normalized
            Numpy array of the same shape as input
    """
    assert ('float' in rgb_norm.dtype.name), "Input image must be color-normalized and hence be float"

    if rgb_norm.ndim == 3:
        # HxWx3
        for i in range(3):
            rgb_norm[:, :, i] = rgb_norm[:, :, i] * std_rgb[i] + mean_rgb[i]
    elif rgb_norm.ndim == 4:
        # Nx3xHxW
        for i in range(3):
            rgb_norm[:, i, :, :] = rgb_norm[:, i, :, :] * std_rgb[i] + mean_rgb[i]
    else:
        raise NotImplementedError(rgb_norm.ndim)

    return rgb_norm


def binarize(im, thres, gt_is_1=True):
    """
    Binarize image

    Args:
        im: Image to binarize
            Numpy array
        thres: Threshold
            Float
        gt_is_1: Whether 1 is for "greater than" or "less than or equal to"
            Boolean
            Optional; defaults to True

    Returns:
        im_bin: Binarized image consisting of only 0's and 1's
            Numpy array of the same shape as input
    """
    if gt_is_1:
        ind_for_1 = im > thres
    else:
        ind_for_1 = im <= thres

    ind_for_0 = np.logical_not(ind_for_1)

    im_bin = deepcopy(im)
    im_bin[ind_for_1] = 1
    im_bin[ind_for_0] = 0

    return im_bin


def get_bbox(mask_0to1, th=0.95):
    indh, indw = np.where(mask_0to1 > th)
    tl_h = np.min(indh)
    tl_w = np.min(indw)
    br_h = np.max(indh)
    br_w = np.max(indw)
    return [tl_w, tl_h, br_w, br_h]


def crop(img, img_bbox, out_size, pad, pad_zero=True, kps=None):
    y1, x1, y2, x2 = img_bbox
    w, h = img.shape[1], img.shape[0]
    x_mid = (x1 + x2) / 2.
    y_mid = (y1 + y2) / 2.
    l = max(x2 - x1, y2 - y1) * out_size / (out_size - 2. * pad)
    x1 = int(np.round(x_mid - l / 2.))
    x2 = int(np.round(x_mid + l / 2.))
    y1 = int(np.round(y_mid - l / 2.))
    y2 = int(np.round(y_mid + l / 2.))
    if kps is not None:
        kps[:, 0] = kps[:, 0] - (y_mid - l / 2.)
        kps[:, 1] = kps[:, 1] - (x_mid - l / 2.)
    b_x = 0
    if x1 < 0:
        b_x = -x1
        x1 = 0
    b_y = 0
    if y1 < 0:
        b_y = -y1
        y1 = 0
    a_x = 0
    if x2 >= h:
        a_x = x2 - (h - 1)
        x2 = h - 1
    a_y = 0
    if y2 >= w:
        a_y = y2 - (w - 1)
        y2 = w - 1
    pad_style = {
        'mode': 'constant',
        'constant_values': 0
    } if pad_zero else {
        'mode': 'edge'
    }
    if img.ndim == 2:
        img_crop = np.pad(
            img[x1:(x2 + 1), y1:(y2 + 1)],
            ((b_x, a_x), (b_y, a_y)),
            **pad_style
        )
    else:
        img_crop = np.pad(
            img[x1:(x2 + 1), y1:(y2 + 1)],
            ((b_x, a_x), (b_y, a_y), (0, 0)),
            **pad_style
        )
    if kps is not None:
        kps[:, 0] = kps[:, 0] * out_size / l
        kps[:, 1] = kps[:, 1] * out_size / l
    return cv2.resize(img_crop, (out_size, out_size))


def sample_vertex_from_mesh(vertex, facet, rnd_idxs=None, u=None, v=None, num_samples=2048):
    # mean = np.mean(vertex, axis=0, keepdims=True)
    # norm = np.max(np.linalg.norm(vertex - mean, axis=1))

    triangles = np.take(vertex, facet, axis=0)
    vx, vy, vz = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(vy - vx, vz - vx), axis=1)
    probs = triangle_areas / np.sum(triangle_areas)

    if rnd_idxs is None:
        rnd_idxs = np.random.choice(np.arange(probs.shape[0]), size=num_samples, p=probs)
    vx, vy, vz = vx[rnd_idxs], vy[rnd_idxs], vz[rnd_idxs]
    if u is None:
        u = np.random.rand(vx.shape[0], 1)
    if v is None:
        v = np.random.rand(vx.shape[0], 1)
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)
    pts = (vx * u + vy * v + vz * w)

    # pts = pts - mean
    # pts = pts / norm
    return pts, rnd_idxs, u, v