import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsrse.models.sr_model import SRModel_fft
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
import torch
from torchvision.transforms.functional import perspective as perspective_transform
from torchvision.transforms.functional import rotate, resize, center_crop
from torchvision.transforms.functional import InterpolationMode
from typing import Sequence, Optional, Union, Tuple
from realesrgan.archs.unet import Unet
from realesrgan.motion_simulation import add_rician_noise, generate_random_mask, random_motion_transform, kspace_scan

@MODEL_REGISTRY.register()
class RealESRNetModel(SRModel_fft):
    """RealESRNet Model for Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealESRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """
        if self.is_train and self.opt.get('high_order_degradation', True):
            # 有空嵌入下面的代码
            rot90_prob= self.opt['rot90_prob'] # 旋转90概率
            undersample_prob= self.opt['undersample_prob'] # 采样概率
            center_fraction_range= self.opt['center_fraction_range'] # 中心分数范围
            acceleration_range= self.opt['acceleration_range'] # 加速度范围
            horizontal_mask_prob = self.opt['horizontal_mask_prob'] # 水平mask概率
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.gt = self.gt[:, 0, :, :]  # only use the Y channel
            self.gt = self.gt.unsqueeze(1)  # add channel dim
            # print(self.gt.size())
            # USM sharpen the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)
            # rotate 90
            if np.random.uniform(0, 1) < rot90_prob:
                self.gt = torch.rot90(self.gt, 1, [2, 3])

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)
            opt_step = np.random.choice(self.opt['step'])
            ori_h, ori_w = self.gt.size()[2:4]

            width, height = self.gt.size()[-1], self.gt.size()[-2]
            #print(self.gt.size())
            # ----------------------- The motion process ----------------------- #

            # 转为单通道灰度图
            L_gt = self.gt.mean(dim=1, keepdim=False)

            K_data = np.zeros((L_gt.shape[0], L_gt.shape[1], L_gt.shape[2]), dtype=np.complex64)
            K_data = torch.from_numpy(K_data).to(self.device)
            width, height = L_gt.size()[-1], L_gt.size()[-2]
            if self.opt['Gp'] == 'v':
                rounds = height
            elif self.opt['Gp'] == 'h':
                rounds = width
            else:
                raise ValueError('Gp should be either "v" or "h"')

            for i in range(0, rounds, opt_step):
                if i > rounds * (31/64) and i < rounds * (33/64):
                    K_data = kspace_scan(L_gt, K_data, i, step=opt_step, Gp=self.opt['Gp'])
                else:
                    out_image = random_motion_transform(L_gt, width, height, rotate_prob=self.opt['rotate_prob'], rotate_range=self.opt['rotate_range'], translation_prob=self.opt['translation_prob'], translation_range=self.opt['translation_range'], perspective_prob=self.opt['perspective_prob'], perspective_range=self.opt['perspective_range'], stretch_prob=self.opt['stretch_prob'], stretch_range=self.opt['stretch_range'])

                    #out_image = center_crop(out_image, (400, 400))
                    K_data = kspace_scan(out_image, K_data, i, step=opt_step, Gp=self.opt['Gp'])
            K_data[:, width * (25//64):width * (39//64), :] = L_gt[:, width * (25//64):width * (39//64), :]

            # ----------------------- The noise process ----------------------- #

            if np.random.uniform(0, 1) < self.opt['rician_noise_prob']:
                temp_reconstructed_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(K_data, dim=(-2, -1)), dim=(-2, -1)))
                rician_std = np.random.uniform(self.opt['rician_noise_range'][0], self.opt['rician_noise_range'][1])
                temp_rician_image = add_rician_noise(temp_reconstructed_image, std=rician_std)

                """ ↑ full image ↑ """

                K_data = torch.fft.fft2(temp_rician_image, dim=(-2, -1))
                K_data = torch.fft.fftshift(K_data, dim=(-2, -1))

            self.undersampled = False

            # ----------------------- The undersample process ----------------------- #

            if np.random.uniform(0, 1) < undersample_prob:
                # center_fraction = np.random.uniform(center_fraction_range[0], center_fraction_range[1])
                acceleration = np.random.choice(acceleration_range)

                center_fraction = 4 / acceleration * 0.08
                mask = generate_random_mask([center_fraction], [acceleration], K_data.shape[-1])
                # print(f"Center Fraction: {center_fraction}, Acceleration: {acceleration}", K_data.shape[-1])
                mask = mask.to(self.device)
                if np.random.uniform(0, 1) > horizontal_mask_prob:
                    mask = mask.t()
                K_data = K_data * mask

            out = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(K_data, dim=(-2, -1)), dim=(-2, -1)))
            out = torch.unsqueeze(out, dim=1)

            # 增加通道数
            out = out.repeat(1, self.opt['network_g']['num_in_ch'], 1, 1)

            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            """ ↑ under image ↑ """

            # random crop
            gt_size = self.opt['gt_size']

            self.gt = self.gt.repeat(1, self.opt['network_g']['num_out_ch'], 1, 1)
            #self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])
            #print(self.lq.size())

            # training pair pool
            self._dequeue_and_enqueue()
            # 模型输出保持与GT一致，便于计算PSNR和SSIM
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        else:
            self.lq = data['lq'].to(self.device)
            self.lq = self.lq[:, 0, :, :]  # only use the Y channel
            self.lq = self.lq.unsqueeze(1)  # add channel dim
            self.lq = self.lq.repeat(1, self.opt['network_g']['num_in_ch'], 1, 1)
            '''
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
                return mask
            # for paired training or validation
            mask = generate_random_mask([0.08], [4], self.lq.size()[-1])
            mask = mask.to(self.device)
            lowfreq_mask = generate_random_mask([0.08], [1/0.08], self.lq.size()[-1])
            lowfreq_mask = lowfreq_mask.to(self.device)

            if np.random.uniform(0, 1) > self.opt['horizontal_mask_prob']:
                mask = mask.t()
                lowfreq_mask = lowfreq_mask.t()

            lq_kdata = torch.fft.fft2(self.lq, dim=(-2, -1))
            lq_kdata = torch.fft.fftshift(lq_kdata, dim=(-2, -1))
            lq_kdata = lq_kdata * mask
            lowfreq_lq_kdata = lq_kdata * lowfreq_mask
            all_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(lq_kdata, dim=(-2, -1)), dim=(-2, -1)))
            lowfreq_image = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(lowfreq_lq_kdata, dim=(-2, -1)), dim=(-2, -1)))

            # 增加通道维度
            # out = torch.unsqueeze(out, dim=1)
            #self.mask = mask # 保存mask
            #self.nmask = torch.logical_not(mask)

            # 增加通道维度
            # out = torch.unsqueeze(out, dim=1)
            #print(all_image.size())
            if self.opt['low_freq'] is True:
                self.lq = torch.stack([all_image, lowfreq_image], dim=1)
            else:
                self.lq = torch.stack([all_image, all_image], dim=1)
            '''
            #print(self.lq.size())
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt = self.gt[:, 0, :, :]  # only use the Y channel
                self.gt = self.gt.unsqueeze(1)  # add channel dim
                # print(self.gt.size())
                self.gt_usm = self.usm_sharpener(self.gt)
                # 模型输出保持与GT一致，便于计算PSNR和SSIM
                self.gt = self.gt.repeat(1, self.opt['network_g']['num_out_ch'], 1, 1)
        '''
        import datetime
        import os
        import torchvision.transforms as transforms

        # Assuming self.lq and self.gt are PyTorch tensors with shape (batch_size, channels, height, width)
        batch_size = self.lq.size(0)

        # Get current time
        current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Create a folder to save images
        folder_path = f"/kaggle/working/images_{current_time}"
        os.makedirs(folder_path, exist_ok=True)

        for sample_index in range(batch_size):
            # Convert to PIL Image
            lq_image = transforms.ToPILImage()(self.lq[sample_index].cpu())
            gt_image = transforms.ToPILImage()(self.gt[sample_index].cpu())

            # Save image with current time and index as filename
            save_path = os.path.join(folder_path, f"lq_image_{current_time}_{sample_index}.png")
            save_path2 = os.path.join(folder_path, f"gt_image_{current_time}_{sample_index}.png")
            lq_image.save(save_path)
            gt_image.save(save_path2)

            print(f"Image saved at: {save_path}")
            print(f"Image saved at: {save_path2}")

        print(f"All images saved in folder: {folder_path}")
        '''
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True