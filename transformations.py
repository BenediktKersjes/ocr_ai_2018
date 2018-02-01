import numpy as np
import PIL
import scipy.ndimage as ndi


def correct_rotation(img):
    """
    Rotate the image, so that it is correctly aligned
    :param img: Pillow image
    :return: Rotated Pillow image
    """
    img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img = img.transpose(PIL.Image.ROTATE_90)
    return img


def random_transform(img, max_angle=20, max_shift=3, max_noise=20):
    """
    Add random transformations to the image
    :param img: Pillow image
    :param max_angle: Maximum value of a rotation in degree
    :param max_shift: Maximum shift in x and y direction
    :param max_noise: Maximum value of noise
    :return: Transformed Pillow image
    """
    img = img.convert('L')
    img = np.asarray(img)

    # Rotate
    angle = float(max_angle * (2 * np.random.rand() - 1))
    img = ndi.rotate(img, angle, reshape=False)

    # Shift
    shift = max_shift * (2 * np.random.rand(2) - 1)
    img = ndi.shift(img, shift)

    # Noise
    noise = max_noise * (2 * np.random.rand(img.shape[1], img.shape[0]) - 1)
    img = np.add(img, noise)

    return PIL.Image.fromarray(img)