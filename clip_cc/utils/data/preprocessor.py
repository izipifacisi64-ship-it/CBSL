from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid, clothid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index, clothid

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return [img_1, img_2], fname, pid, camid, index


class MaskPreprocessor(Dataset):
    """
    带 parsing mask 的 Preprocessor。

    mask 查找顺序:
      1. mask_dir/pid_dir/stem.npy  (PRCC 子目录结构)
      2. mask_dir/stem.npy          (扁平结构)

    返回格式与 Preprocessor 完全一致:
      (img, fname, pid, camid, index, clothid)
    """

    def __init__(self, dataset, root, mask_dir,
                 height=256, width=128,
                 normalizer=None,
                 parsing_eraser=None,
                 random_erasing=None,
                 flip_prob=0.5,
                 pad=10):
        super().__init__()
        self.dataset = dataset
        self.root = root
        self.mask_dir = mask_dir
        self.height = height
        self.width = width
        self.normalizer = normalizer
        self.parsing_eraser = parsing_eraser
        self.random_erasing = random_erasing
        self.flip_prob = flip_prob
        self.pad = pad

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid, clothid = self.dataset[index]

        fpath = osp.join(self.root, fname) if self.root else fname
        img = Image.open(fpath).convert('RGB')

        # 加载 mask
        pid_dir = os.path.basename(os.path.dirname(fpath))
        stem = os.path.splitext(os.path.basename(fpath))[0]
        mask = None
        mask_path = osp.join(self.mask_dir, pid_dir, stem + '.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
        else:
            mask_path = osp.join(self.mask_dir, stem + '.npy')
            if os.path.exists(mask_path):
                mask = np.load(mask_path)

        # 同步空间变换
        img, mask = self._sync_spatial(img, mask)

        # ToTensor + Normalize
        from torchvision.transforms import functional as TF
        img = TF.to_tensor(img)
        if self.normalizer is not None:
            img = self.normalizer(img)

        # 通用 RandomErasing
        if self.random_erasing is not None:
            img = self.random_erasing(img)

        # ParsingGuidedErasing
        if self.parsing_eraser is not None and mask is not None:
            _, h, w = img.shape
            if mask.shape[0] != h or mask.shape[1] != w:
                mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))
            img = self.parsing_eraser(img, mask)

        return img, fname, pid, camid, index, clothid

    def _sync_spatial(self, img, mask):
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode

        h, w = self.height, self.width

        img = TF.resize(img, (h, w), interpolation=InterpolationMode.BICUBIC)
        if mask is not None:
            mask = np.array(Image.fromarray(mask).resize((w, h), Image.NEAREST))

        if random.random() < self.flip_prob:
            img = TF.hflip(img)
            if mask is not None:
                mask = np.ascontiguousarray(mask[:, ::-1])

        if self.pad > 0:
            img = TF.pad(img, self.pad)
            if mask is not None:
                mask = np.pad(mask, self.pad, mode='constant', constant_values=0)

        padded_h, padded_w = img.size[1], img.size[0]
        top = random.randint(0, padded_h - h)
        left = random.randint(0, padded_w - w)
        img = TF.crop(img, top, left, h, w)
        if mask is not None:
            mask = mask[top:top + h, left:left + w]

        return img, mask