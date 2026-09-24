from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
from torch import nn
import tqdm
from torch.cuda import amp
import torch.nn.functional as F
from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch

from .utils.eval_metrics1 import evaluate_with_clothes


def extract_cnn_feature(model, inputs, cams=None):
    """
    返回:
      global_feat: 全局特征 (B, 1280)
      fused:       全局 + 局部拼接归一化 (B, 1280+768)
    兼容 model 返回 2 个或 3 个值。
    """
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs)
    inputs = inputs.cuda()

    # 兼容返回 2 个值(原版) 或 3 个值(方案C)
    out = model(inputs)
    global_feat = out[0]
    local_feats = out[1]

    global_feat = global_feat.data.cpu()
    local_feats = local_feats.data.cpu()

    local_mean = local_feats.mean(dim=1)

    g_norm = F.normalize(global_feat, dim=1)
    l_norm = F.normalize(local_mean, dim=1)

    fused = torch.cat([g_norm, l_norm], dim=1)
    fused = F.normalize(fused, dim=1)

    return global_feat, fused


def extract_features(model, data_loader):
    """聚类用: 返回全局特征"""
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    with torch.no_grad():
        for i, (imgs, fnames, pids, camids, _, clothid) in enumerate(tqdm.tqdm(data_loader)):
            outputs, _ = extract_cnn_feature(model, imgs, camids)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid
    return features, labels


def extract_features2(model, data_loader):
    """评估用: 返回融合特征"""
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    with torch.no_grad():
        for i, (imgs, fnames, pids, camids, _, clothid) in enumerate(tqdm.tqdm(data_loader)):
            _, outputs = extract_cnn_feature(model, imgs, camids)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid
    return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    return all_cmc, mAP


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]

        query_ids1 = np.asarray([pid for _, pid, _, _ in query])
        query_cams1 = np.asarray([cam for _, _, cam, _ in query])
        query_clothid1 = np.asarray([clothid for _, _, _, clothid in query])
        gallery_ids1 = np.asarray([pid for _, pid, _, _ in gallery])
        gallery_cams1 = np.asarray([cam for _, _, cam, _ in gallery])
        gallery_clothid1 = np.asarray([clothid for _, _, _, clothid in gallery])
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # General 评估
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP (General): {:4.1%}'.format(mAP))

    cmc_general, mAP_general = eval_func(distmat, query_ids1, gallery_ids1, query_cams1, gallery_cams1)
    print('Mean AP (General): {:4.1%}'.format(mAP_general))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_general[k-1]))

    # SC 评估
    cmc_sc, mAP_sc = evaluate_with_clothes(distmat, query_ids1, gallery_ids1, query_cams1, gallery_cams1,
                                           query_clothid1, gallery_clothid1, mode='SC')
    print('SC: Rank-1={:.1%}, mAP={:.1%}'.format(cmc_sc[0], mAP_sc))

    # CC 评估
    cmc_cc, mAP_cc = evaluate_with_clothes(distmat, query_ids1, gallery_ids1, query_cams1, gallery_cams1,
                                           query_clothid1, gallery_clothid1, mode='CC')
    print('CC: Rank-1={:.1%}, mAP={:.1%}'.format(cmc_cc[0], mAP_cc))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_cc[k - 1]))

    if (not cmc_flag):
        # 返回 CC 的 Rank-1 和 mAP（两个值）
        return cmc_cc[0], mAP_cc

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features2(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat,
                               query=query, gallery=gallery, cmc_flag=cmc_flag)
        if (not rerank):
            return results
        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat,
                            query=query, gallery=gallery, cmc_flag=cmc_flag)