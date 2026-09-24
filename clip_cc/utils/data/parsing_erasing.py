# -*- coding: utf-8 -*-
"""
clip_cc/utils/data/parsing_erasing.py

原版课程式衣物擦除。

三种模式:
  clothing_only  : 只擦衣物区域
  clothing_heavy : 擦衣物 + 随机擦一个肢体/脚部区域
  identity_mask  : 只保留头+四肢，其余全擦 (最极端)

课程调度:
  前 30% epoch → 100% clothing_only
  中 30% epoch → 70% clothing_only + 30% clothing_heavy
  后 40% epoch → 40% clothing_only + 40% clothing_heavy + 20% identity_mask
"""
import random
import numpy as np
import torch
from PIL import Image

# 区域常量 (与 generate_masks.py 一致)
REGION_BG       = 0
REGION_CLOTHING = 1
REGION_HEAD     = 2
REGION_LIMBS    = 3
REGION_FEET     = 4

# ImageNet 归一化后的均值
FILL_VALS = [0.485, 0.456, 0.406]


class ParsingGuidedErasing:

    def __init__(self, probability=0.5, mode='curriculum',
                 current_epoch=0, total_epochs=30):
        self.p = probability
        self.mode = mode
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

    @property
    def _progress(self):
        return self.current_epoch / max(self.total_epochs, 0.5)

    def _select_mode(self):
        if self.mode != 'curriculum':
            return self.mode
        p = self._progress
        r = random.random()
        if p < 0.3:
            return 'clothing_only'
        elif p < 0.6:
            return 'clothing_only' if r < 0.7 else 'clothing_heavy'
        else:
            if r < 0.4:
                return 'clothing_only'
            elif r < 0.8:
                return 'clothing_heavy'
            else:
                return 'identity_mask'

    @staticmethod
    def _fill_region(img, mask_tensor, region_ids, alpha=0.5):
        """软填充: alpha×均值 + (1-alpha)×原图"""
        combined = torch.zeros(mask_tensor.shape, dtype=torch.bool)
        for rid in region_ids:
            combined |= (mask_tensor == rid)
        if combined.any():
            for c in range(3):
                orig = img[c][combined]
                fill = torch.full_like(orig, FILL_VALS[c])
                img[c][combined] = alpha * fill + (1 - alpha) * orig

    def __call__(self, img, mask):
        """
        Args:
            img:  Tensor [C, H, W], 已 normalize
            mask: np.ndarray [H, W], 值 0~4
        Returns:
            img:  Tensor [C, H, W]
        """
        if random.random() > self.p:
            return img

        # 尺寸对齐
        _, h, w = img.shape
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))

        mt = torch.from_numpy(mask.astype(np.int64))
        mode = self._select_mode()

        if mode == 'clothing_only':
            self._fill_region(img, mt, [REGION_CLOTHING])

        elif mode == 'clothing_heavy':
            self._fill_region(img, mt, [REGION_CLOTHING])
            self._fill_region(img, mt, [random.choice([REGION_LIMBS, REGION_FEET])])

        elif mode == 'identity_mask':
            keep = (mt == REGION_HEAD) | (mt == REGION_LIMBS)
            erase = ~keep
            if erase.any():
                for c in range(3):
                    orig = img[c][erase]
                    fill = torch.full_like(orig, FILL_VALS[c])
                    img[c][erase] = 0.5 * fill + 0.5 * orig

        return img

    def __repr__(self):
        return f'ParsingGuidedErasing(p={self.p}, mode={self.mode}, progress={self._progress:.2f})'