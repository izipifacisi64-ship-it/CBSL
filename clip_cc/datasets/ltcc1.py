import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
from scipy.io import loadmat

from ..utils.data import BaseImageDataset


# from tools.utils import mkdir_if_missing, write_json, read_json


class LTCC(BaseImageDataset):
    """ LTCC

    Reference:
        Qian et al. Long-Term Cloth-Changing Person Re-identification. arXiv:2005.12633, 2020.

    URL: https://naiq.github.io/LTCC_Perosn_ReID.html#
    """
    dataset_dir = 'LTCC_ReID'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self._check_before_run()

        train = self._process_dir_train(self.train_dir)
        query, gallery = self._process_dir_test(self.query_dir, self.gallery_dir)

        if verbose:
            print("=> LTCC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        # label2pid = {label: pid for label, pid in enumerate(pid_container)}
        #
        # np.save('/home/ykding/logs/ltcc/idx_to_labels.npy', pid2label)
        # np.save('/home/ykding/logs/ltcc/label_to_pid.npy', label2pid)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))

        # files = os.listdir(dir)

        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes = pattern2.search(img_path).group(1)
            camid -= 1  # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]

            #############
            file_name = os.path.basename(img_path)
            seg_path = os.path.join(dir_path+'-mask',file_name)

            # dataset.append((img_path, pid, camid, clothes_id,seg_path))
            dataset.append((img_path, pid, camid, clothes_id))
            # dataset.append((img_path, pid, camid))

            pid2clothes[pid, clothes_id] = 1

        num_imgs = len(dataset)

        return dataset

    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.png'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.png'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')
        pattern2 = re.compile(r'(\w+)_c')

        pid_container = set()
        clothes_container = set()
        for img_path in query_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1  # index starts from 0
            clothes_id = clothes2label[clothes_id]
            file_name = os.path.basename(img_path)
            seg_path = os.path.join(query_path + '-mask', file_name)

            # query_dataset.append((img_path, pid, camid, clothes_id,seg_path))
            query_dataset.append((img_path, pid, camid, clothes_id))

            # query_dataset.append((img_path, pid, camid))


        for img_path in gallery_img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            clothes_id = pattern2.search(img_path).group(1)
            camid -= 1  # index starts from 0
            clothes_id = clothes2label[clothes_id]

            file_name = os.path.basename(img_path)
            seg_path = os.path.join(gallery_path + '-mask', file_name)

            # gallery_dataset.append((img_path, pid, camid, clothes_id,seg_path))

            gallery_dataset.append((img_path, pid, camid, clothes_id))
            # gallery_dataset.append((img_path, pid, camid))

        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset

