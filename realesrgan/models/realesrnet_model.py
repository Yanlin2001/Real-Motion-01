import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
import torch
from torchvision.transforms.functional import perspective as perspective_transform
from torchvision.transforms.functional import rotate, resize, center_crop
from torchvision.transforms.functional import InterpolationMode

@MODEL_REGISTRY.register()
class RealESRNetModel(SRModel):
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
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM sharpen the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            width, height = self.gt.size()[-1], self.gt.size()[-2]
            # ----------------------- The first motion process ----------------------- #
            def random_motion_transform(image, width, height,
                            rotate_prob = 0.5, rotate_range = [-5, 5],
                            translation_prob = [0.2, 0.2, 0.6], translation_range = [-0.1, 0.1],
                            perspective_prob = [0.3, 0.3, 0.4], perspective_range = [-0.1, 0.1],
                            stretch_prob = 0.4, stretch_range = [-0.1, 0.1]):
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


            def kspace_scan(image_tensor, K_data, cur_round, tol_round):
                # 将图像张量转换为K空间数据
                k_space_data = torch.fft.fft2(image_tensor, dim=(-2, -1))
                print('B01')
                # 进行 fftshift 操作将低频移到中心
                k_space_data = torch.fft.fftshift(k_space_data, dim=(-2, -1))
                print('B02')
                # 获取图像的高度和宽度
                _, H, W = image_tensor.shape

                # 计算当前应该填充的行范围
                start_row = cur_round * H // tol_round
                end_row = (cur_round + 1) * H // tol_round
                print(start_row, end_row)
                print('B03')
                # 截取并填充到K_data中
                K_data[:, start_row:end_row, :] = k_space_data[:, start_row:end_row, :]
                print('B04')
                return K_data

            print('gt shape:', self.gt.shape)

            # 转为单通道灰度图
            L_gt = self.gt.mean(dim=1, keepdim=False)

            print('L_gt shape:', L_gt.shape)

            rounds = 5
            K_data = np.zeros((1, L_gt.shape[1], L_gt.shape[2]), dtype=np.complex64)
            print('before rounds')
            for i in range(rounds):
                if i == rounds//2:
                    K_data = kspace_scan(L_gt, K_data, i, rounds)
                else:
                    out_image = random_motion_transform(L_gt, width, height)
                    print('A01')
                    #out_image = center_crop(out_image, (400, 400))
                    K_data = kspace_scan(out_image, K_data, i, rounds)
                    print('A02')
            print('after rounds')
            out = np.abs(np.fft.ifft2(np.fft.ifftshift(K_data, axes=(-2, -1)), axes=(-2, -1)))

            print('out shape:', out.shape)
            # 增加通道维度
            out = np.expand_dims(out, axis=1)
            print('out shape:', out.shape)
            # clamp and round
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            #self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt['scale'])

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            # for paired training or validation
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealESRNetModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True