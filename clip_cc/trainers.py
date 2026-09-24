from __future__ import print_function, absolute_import

import time
from torch.cuda import amp
from .utils.meters import AverageMeter
import torch
import torch.nn.functional as F
from datetime import datetime


def local_contrastive_loss(local_feats, labels):
    B, num_regions, D = local_feats.shape
    local_feats = F.normalize(local_feats, dim=-1)

    same_id_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    triu_mask = torch.triu(torch.ones_like(same_id_mask, dtype=torch.bool), diagonal=1)
    pos_mask = same_id_mask & triu_mask

    loss_per_region = []
    for k in range(num_regions):
        feats_k = local_feats[:, k, :]
        sim_matrix = torch.mm(feats_k, feats_k.t())
        pos_sim = sim_matrix[pos_mask]
        if pos_sim.numel() > 0:
            loss_k = (1 - pos_sim).mean()
        else:
            loss_k = torch.tensor(0.0, device=local_feats.device)
        loss_per_region.append(loss_k)

    loss = sum(loss_per_region) / num_regions
    return loss


def diversity_loss(local_feats):
    """多样性正则：鼓励不同区域特征尽量不同，防止塌缩"""
    B, G, D = local_feats.shape
    feats = F.normalize(local_feats, dim=-1)
    sim = torch.bmm(feats, feats.transpose(1, 2))
    mask = ~torch.eye(G, dtype=torch.bool, device=sim.device)
    mask = mask.unsqueeze(0).expand(B, -1, -1)
    loss = sim[mask].mean()
    return loss


class VITFP16(object):
    def __init__(self, encoder, memory=None, lambda_local=0.7, lambda_div=0.1):
        super(VITFP16, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.lambda_local = lambda_local
        self.lambda_div = lambda_div

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        scaler = amp.GradScaler()
        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            with amp.autocast(enabled=True):
                inputs, labels, indexes = self._parse_data(inputs)
                global_feat, local_feats = self.encoder(inputs)

                loss_global = self.memory(global_feat, labels)
                loss_local = local_contrastive_loss(local_feats, labels)
                loss_div = diversity_loss(local_feats)

                loss = 1 * loss_global + self.lambda_local * loss_local + self.lambda_div * loss_div

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Div {:.3f}\t'
                      .format(epoch + 1, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              loss_div.item()))
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes, clothid = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()