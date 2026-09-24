# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import warnings
warnings.filterwarnings('ignore')
import os.path as osp
import random
import numpy as np
import sys
import os
import collections
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from clip_cc import datasets
from clip_cc.models.cm import ClusterMemory
from clip_cc.trainers import VITFP16
from clip_cc.evaluators import Evaluator, extract_features
from clip_cc.utils.data import IterLoader
from clip_cc.utils.data import transforms as T
from clip_cc.utils.data.preprocessor import Preprocessor, MaskPreprocessor
from clip_cc.utils.data.parsing_erasing import ParsingGuidedErasing
from clip_cc.utils.logging import Logger
from clip_cc.utils.serialization import load_checkpoint, save_checkpoint
from clip_cc.utils.faiss_rerank import compute_jaccard_distance
from clip_cc.utils.data.sampler import RandomMultipleGallerySampler
from clip_cc.models.model_clip import make_model
from clip_cc.utils.prepare_optimizer import make_vit_optimizer
from clip_cc.utils.prepare_scheduler import create_scheduler
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clip_cc.utils.data.parsing_erasing import REGION_BG, REGION_CLOTHING, REGION_HEAD, REGION_LIMBS, REGION_FEET
from PIL import Image

start_epoch = best_mAP = 0


# =============================================================================
# 可视化
# =============================================================================

def visualize_parsing_erasing(dataset, mask_dir, save_path, n_samples=4):
    from torchvision.transforms import functional as TF

    all_items = sorted(dataset.train)
    step = max(1, len(all_items) // n_samples)
    samples = [all_items[i * step] for i in range(n_samples)]

    region_colors = {
        REGION_BG:       [0, 0, 0, 0],
        REGION_CLOTHING: [255, 80, 80, 140],
        REGION_HEAD:     [80, 180, 255, 140],
        REGION_LIMBS:    [80, 255, 120, 140],
        REGION_FEET:     [255, 200, 60, 140],
    }

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denormalize(tensor):
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

    def make_overlay(img_np, mask):
        o = img_np.copy().astype(np.float32)
        for rid, c in region_colors.items():
            if rid == REGION_BG:
                continue
            m = (mask == rid)
            if m.any():
                a = c[3] / 255.0
                for ch in range(3):
                    o[:, :, ch] = np.where(m, o[:, :, ch] * (1 - a) + c[ch] * a, o[:, :, ch])
        return np.clip(o, 0, 255).astype(np.uint8)

    def apply_erasing(img_pil, mask, mode):
        from torchvision.transforms import Normalize
        normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_resized = TF.resize(img_pil, (256, 128), interpolation=InterpolationMode.BICUBIC)
        img_tensor = normalizer(TF.to_tensor(img_resized))
        mask_resized = np.array(Image.fromarray(mask).resize((128, 256), Image.NEAREST))
        eraser = ParsingGuidedErasing(probability=1.0, mode=mode, current_epoch=0, total_epochs=60)
        return denormalize(eraser(img_tensor.clone(), mask_resized))

    def load_mask(img_full_path):
        pid_dir = osp.basename(osp.dirname(img_full_path))
        stem = osp.splitext(osp.basename(img_full_path))[0]
        mask_path = osp.join(mask_dir, pid_dir, stem + '.npy')
        if not osp.exists(mask_path):
            mask_path = osp.join(mask_dir, stem + '.npy')
        if osp.exists(mask_path):
            return np.load(mask_path)
        return None

    fig, axes = plt.subplots(n_samples, 5, figsize=(15, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'Parsing mask', 'clothing_only', 'clothing_heavy', 'identity_mask']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight='bold')

    for i, item in enumerate(samples):
        img_pil = Image.open(item[0]).convert('RGB')
        mask = load_mask(item[0])
        if mask is None:
            mask = np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8)
            print(f'  [WARN] mask not found: {osp.basename(item[0])}')

        img_np = np.array(img_pil.resize((128, 256), Image.BICUBIC))
        axes[i, 0].imshow(img_np)

        mask_resized = np.array(Image.fromarray(mask).resize((128, 256), Image.NEAREST))
        axes[i, 1].imshow(make_overlay(img_np, mask_resized))

        for j, mode in enumerate(['clothing_only', 'clothing_heavy', 'identity_mask']):
            axes[i, 2 + j].imshow(apply_erasing(img_pil, mask, mode))

        for j in range(5):
            axes[i, j].axis('off')

    fig.text(0.5, 0.01,
             'Red=Clothing  Blue=Head  Green=Limbs  Yellow=Feet',
             ha='center', fontsize=11, color='gray', style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Visualize] Saved to: {save_path}')


# =============================================================================
# 工具函数
# =============================================================================

def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir))


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, parsing_eraser=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    random_erasing = T.RandomErasing(probability=0.3, mean=[0.485, 0.456, 0.406])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    sampler = RandomMultipleGallerySampler(train_set, num_instances)

    return IterLoader(
        DataLoader(
            MaskPreprocessor(
                dataset=train_set,
                root=dataset.images_dir,
                mask_dir=args.mask_dir,
                height=height, width=width,
                normalizer=normalizer,
                parsing_eraser=parsing_eraser,
                random_erasing=random_erasing,
                flip_prob=0.5, pad=10,
            ),
            batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=False, pin_memory=True, drop_last=True
        ), length=iters)


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(), normalizer
    ])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    return DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)


def create_model(args):
    model = make_model()
    model.cuda()
    return nn.DataParallel(model)


def adaptive_eps(dist_matrix, rho=0.005):
    tri_mat = dist_matrix[np.triu_indices(dist_matrix.shape[0], 1)]
    tri_mat_sorted = np.sort(tri_mat)
    top_num = int(np.round(rho * len(tri_mat_sorted)))
    return tri_mat_sorted[:top_num].mean()


def multi_scale_dbscan(dist_matrix, eps_values, min_samples=4):
    all_labels = []
    for eps in eps_values:
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
        all_labels.append(cluster.fit_predict(dist_matrix))

    all_labels = np.array(all_labels)
    final_labels = -1 * np.ones(all_labels.shape[1], dtype=int)

    for i in range(all_labels.shape[1]):
        labels_i = all_labels[:, i]
        valid = labels_i[labels_i != -1]
        if len(valid) == 0:
            final_labels[i] = -1
        elif len(set(valid)) == 1:
            final_labels[i] = valid[0]
        else:
            nearest = np.argsort(dist_matrix[i])[:min_samples]
            neighbor = all_labels[:, nearest].flatten()
            neighbor = neighbor[neighbor != -1]
            final_labels[i] = collections.Counter(neighbor).most_common(1)[0][0] if len(neighbor) > 0 else -1
    return final_labels


def relabel_pseudo_labels(labels):
    valid = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(valid)}
    return np.array([mapping[l] if l != -1 else -1 for l in labels]), len(mapping)


# =============================================================================
# main
# =============================================================================

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = False
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    assert os.path.isdir(args.mask_dir), \
        f'mask 目录不存在: {args.mask_dir}\n请先运行 generate_masks.py'

    iters = args.iters if (args.iters > 0) else None
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    model = create_model(args)
    evaluator = Evaluator(model)
    optimizer = make_vit_optimizer(model)
    lr_scheduler = create_scheduler(optimizer)
    trainer = VITFP16(model, lambda_div=args.lambda_div)

    parsing_eraser = ParsingGuidedErasing(
        probability=args.erase_prob,
        mode='curriculum',
        total_epochs=args.epochs,
    )
    print(f'[Eraser] ParsingGuidedErasing (mask_dir={args.mask_dir})')

    vis_path = osp.join(args.logs_dir, 'parsing_erasing_visualization.png')
    os.makedirs(args.logs_dir, exist_ok=True)
    visualize_parsing_erasing(dataset, args.mask_dir, vis_path, n_samples=4)

    best_mAP = 0.0

    for epoch in range(args.epochs):
        print('=> EPOCH num={}'.format(epoch + 1))
        parsing_eraser.current_epoch = epoch
        print(f'[Eraser] {parsing_eraser}')

        with torch.no_grad():
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print('==> Extract features for clustering...')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers,
                                             testset=sorted(dataset.train))
            features, _ = extract_features(model, cluster_loader)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if args.dataset.lower() == 'prcc':
                eps_values = [adaptive_eps(rerank_dist, rho=r) for r in [0.002, 0.004, 0.006]]
                print(f'Multi-scale eps for prcc: {eps_values}')
                pseudo_labels = multi_scale_dbscan(rerank_dist, eps_values, min_samples=5)
            else:
                eps_small = adaptive_eps(rerank_dist, rho=0.003)
                eps_large = adaptive_eps(rerank_dist, rho=0.006)
                print(f'Multi-scale eps: eps_small={eps_small:.3f}, eps_large={eps_large:.3f}')
                pseudo_labels = multi_scale_dbscan(rerank_dist, [eps_small, eps_large], min_samples=4)

            pseudo_labels, num_cluster = relabel_pseudo_labels(pseudo_labels)
            num_outliers = (pseudo_labels == -1).sum()
            print(f'Clusters: {num_cluster}, Outliers: {num_outliers} '
                  f'({num_outliers / len(pseudo_labels):.2%})')

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[label].append(features[i])
            return torch.stack([torch.stack(centers[idx]).mean(0) for idx in sorted(centers.keys())])

        cluster_features = generate_cluster_features(pseudo_labels, features)
        del cluster_loader, features

        memory = ClusterMemory(1280, num_cluster, temp=args.temp, momentum=args.momentum).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        trainer.memory = memory

        pseudo_labeled_dataset = [(fname, label.item(), cid, clothid)
                                  for (fname, _, cid, clothid), label in zip(sorted(dataset.train), pseudo_labels)
                                  if label != -1]

        if len(pseudo_labeled_dataset) < args.batch_size:
            print(f'WARNING: only {len(pseudo_labeled_dataset)} samples, skip.')
            lr_scheduler.step()
            torch.cuda.empty_cache()
            continue

        train_loader = get_train_loader(
            args, dataset, args.height, args.width,
            args.batch_size, args.workers,
            num_instances=16, iters=iters,
            trainset=pseudo_labeled_dataset,
            parsing_eraser=parsing_eraser,
        )

        print(f'=> Current Lr: {optimizer.param_groups[0]["lr"]:.2e}')

        time.sleep(0.5)
        train_loader.new_epoch()
        time.sleep(0.5)

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            rank1, mAP_cc = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (rank1 > best_mAP)
            best_mAP = max(rank1, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model.pth.tar'))

            print('\n * Finished epoch {:3d}  Rank-1: {:5.1%}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch + 1, rank1, mAP_cc, best_mAP, ' *' if is_best else ''))
        lr_scheduler.step()
        torch.cuda.empty_cache()
        print('=> CUDA cache is released.')

    # 10-trial 测试
    print('==> Test with the best model (10-trial average):')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    all_rank1, all_mAP = [], []
    for trial in range(10):
        dataset.resample_gallery(seed=trial)
        tl = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
        print(f'\n--- Trial {trial + 1}/10 ---')
        r1, m = evaluator.evaluate(tl, dataset.query, dataset.gallery, cmc_flag=False)
        all_rank1.append(r1)
        all_mAP.append(m)

    print('\n' + '=' * 50)
    print(f'10-Trial Average:  Rank-1: {np.mean(all_rank1):.1%} +/- {np.std(all_rank1):.1%}')
    print(f'                   mAP:    {np.mean(all_mAP):.1%} +/- {np.std(all_mAP):.1%}')
    print('=' * 50)
    print('Total time:', timedelta(seconds=time.monotonic() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='prcc', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--num-instances', type=int, default=16)
    parser.add_argument('--eps', type=float, default=0.7)
    parser.add_argument('--eps-gap', type=float, default=0.02)
    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--lambda-div', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--erase-prob', type=float, default=0.5)
    parser.add_argument('--mask-dir', type=str, required=True)
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, default="/root")
    parser.add_argument('--logs-dir', type=str, default=osp.join(working_dir, 'logs'))
    main()