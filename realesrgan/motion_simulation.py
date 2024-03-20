import torch
from torchvision.transforms.functional import perspective as perspective_transform
from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from typing import Sequence, Optional, Union, Tuple

''' 添加Rician噪声 '''
def add_rician_noise(image, mean=0, std=25):
    image = image.float()
    # 生成高斯噪声并添加到图像上
    noise_real  = torch.randn_like(image) * std + mean
    noise_imaginary = torch.randn_like(image) * std + mean
    noisy_image = torch.sqrt((image + noise_real)**2 + noise_imaginary**2)
    # 将像素值裁剪到 [0, 1] 范围内
    # noisy_image = torch.clamp(noisy_image, 0, 1)
    # print("std:", std)
    return noisy_image

''' 生成随机掩膜 '''
def generate_random_mask(center_fractions: Sequence[float], accelerations: Sequence[int], num_cols: int, seed: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    if len(center_fractions) != len(accelerations):
        raise ValueError("Number of center fractions should match number of accelerations")

    rng = np.random.RandomState(seed)
    choice = rng.randint(0, len(accelerations))
    center_fraction = center_fractions[choice]
    acceleration = accelerations[choice]

    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)

    mask = rng.uniform(size=num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True

    mask_shape = [1, 1] + [1] * (len(mask.shape) - 2)
    mask_shape[-2] = num_cols
    mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    # print("Generated Random Mask:")
    # print(mask)
    #print(f"Center Fraction: {center_fraction}, Acceleration: {acceleration}")
    true_count = int(mask.sum())
    #print(f"Number of True values in the mask: {true_count}")

    return mask

''' 随机运动伪影 '''
def random_motion_transform(image, width, height,
                            rotate_prob = 0.3, rotate_range = [-1, 1],
                            translation_prob = [0.6, 0.0, 0.4], translation_range = [-0.01, 0.01],
                            perspective_prob = [0.0, 0.0, 1.0], perspective_range = [-0.01, 0.01],
                            stretch_prob = 0.0, stretch_range = [-0.01, 0.01]):
    # 旋转图像 rotate
    if np.random.uniform(0, 1) < rotate_prob:
        rotate_angle = np.random.uniform(rotate_range[0], rotate_range[1])
        out = rotate(image, rotate_angle, interpolation=InterpolationMode.BILINEAR)
    else:
        out = image

    # 平移图像 translation
    translation_type = np.random.choice(["left-right", "up-down", "keep"], p=translation_prob)
    translation_rate = np.random.uniform(-translation_range[0], translation_range[0])
    if translation_type == "left-right":
        out = out.roll(int(translation_rate * width), 2)
    elif translation_type == "up-down":
        out = out.roll(int(translation_rate * height), 1)
    else:
        out = out

    # 透视变换 perspective
    original_pts = [[0, 0], [width, 0], [0, height], [width, height]]
    perspective_type = np.random.choice(["pitch", "yaw", "keep"], p=perspective_prob)
    pitch_rate1 = np.random.uniform(-perspective_range[0], perspective_range[0])
    pitch_rate2 = np.random.uniform(-perspective_range[0], perspective_range[0])
    yaw_rate1 = np.random.uniform(-perspective_range[1], perspective_range[1])
    yaw_rate2 = np.random.uniform(-perspective_range[1], perspective_range[1])
    if np.random.uniform(0, 1) < stretch_prob:
        stretch_rate1 = np.random.uniform(stretch_range[0], stretch_range[1])
        stretch_rate2 = np.random.uniform(stretch_range[0], stretch_range[1])
    else:
        stretch_rate1 = 0
        stretch_rate2 = 0
    if perspective_type == "pitch":
        pitch_pts = [
            [0 - pitch_rate1 * width, 0 - stretch_rate1 * height],
            [width + pitch_rate1 * width, 0 - stretch_rate1 * height],
            [0 - pitch_rate2 * width, height + stretch_rate2 * height],
            [width + pitch_rate2 * width, height + stretch_rate2 * height],
        ]
        yaw_pts = original_pts
        out = perspective_transform(out, original_pts, pitch_pts)
    elif perspective_type == "yaw":
        yaw_pts = [
            [0 - stretch_rate1 * width, 0 - yaw_rate1 * height],
            [width + stretch_rate2 * width, 0 - yaw_rate2 * height],
            [0 - stretch_rate1 * width, height + yaw_rate1 * height],
            [width + stretch_rate2 * width, height + yaw_rate2 * height],
        ]
        pitch_pts = original_pts
        out = perspective_transform(out, original_pts, yaw_pts)
    else:
        pitch_pts = original_pts
        yaw_pts = original_pts
        stretch_pts = [
            [0 , 0 - stretch_rate1 * height],
            [width, 0 - stretch_rate1 * height],
            [0 , height + stretch_rate2 * height],
            [width, height + stretch_rate2 * height],
        ]
        out = perspective_transform(out, original_pts, stretch_pts)

    return out

def kspace_scan(image_tensor, K_data, cur_round, step=2, Gp='v'):

    k_data = torch.fft.fft2(image_tensor, dim=(-2, -1))
    k_data = torch.fft.fftshift(k_data, dim=(-2, -1))

    # 计算当前应该填充的行范围
    start_row = cur_round
    end_row = cur_round + step
    # print(start_row, end_row)

    # 截取并填充到K_data中
    if Gp == 'v':
        K_data[:, start_row:end_row, :] = k_data[:, start_row:end_row, :]
    elif Gp == 'h':
        K_data[:, :, start_row:end_row] = k_data[:, :, start_row:end_row]
    else :
        print('Error: Gp should be either "v" or "h"')

    return K_data


def apply_motion_and_reconstruct(combined_image, rounds=64, Gp = 'v',step = 8):

    K_data = torch.zeros_like(combined_image, dtype=torch.complex128)
    width, height = combined_image.shape[-2], combined_image.shape[-1]
    if Gp == 'v':
        rounds = height
    elif Gp == 'h':
        rounds = width
    else:
        raise ValueError("Gp should be 'v' or 'h'")

    for cur_phase in range(0, rounds, step):
        if cur_phase > rounds * (31/64) and cur_phase < rounds * (33/64):
            K_data = kspace_scan(combined_image, K_data, cur_phase, step=step, Gp=Gp)
        else:
            out_image = random_motion_transform(combined_image, width, height)
            combined_result = out_image
            K_data = kspace_scan(combined_result, K_data, cur_phase, step=step, Gp=Gp)
    combined_k_data = torch.fft.fft2(combined_image, dim=(-2, -1))
    combined_k_data = torch.fft.fftshift(combined_k_data, dim=(-2, -1))
    K_data[:, width * (25//64):width * (39//64), :] = combined_k_data[:, width * (25//64):width * (39//64), :]
    reconstructed_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(K_data, dim=(-2, -1)), dim=(-2, -1)))

    return reconstructed_image


