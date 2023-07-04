import copy
import os
import pickle

import torch

import numpy as np

from tqdm import tqdm

from mmdet.core.bbox.iou_calculators import bbox_overlaps


class JointDataset:
    """combine multiple datasets into one dataset"""
    def __init__(self, datasets):
        self.acc_len = [0]
        self.datasets = datasets
        for ds in datasets:
            self.acc_len.append(self.acc_len[-1] + len(ds))
            
    def __len__(self):
        return self.acc_len[-1]
    
    def __getitem__(self, idx):
        for ds, start, end in zip(self.datasets, self.acc_len[:-1], self.acc_len[1:]):
            if idx < end:
                return ds[idx - start]
            
            
class PseudoVideoDataset:
    """repeat the same image for multiple times to form a pseudo video dataset"""
    def __init__(self, src, clip_sampler) -> None:
        self.ds = src
        self.num_imgs = clip_sampler["num_ref_imgs"]
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        data["ann_info"]["instance_ids"] = np.arange(len(data["ann_info"]["bboxes"]))
        frames = []
        for f in range(self.num_imgs):
            frames.append(copy.deepcopy(data))
            frames[-1]["img_info"]["frame_id"] = f
        return frames
    

class WeightedDataset:
    """sample one datasets with different weights for each sample according to bbox count and iou"""
    def __init__(
        self, 
        dataset, 
        iters_per_epoch=20000,
        iou_multiplier=[(8.0, 2.0)],
        obj_multiplier=[(0.5, 0.5)],
    ):
        self.dataset = dataset
        self.iters_per_epoch = iters_per_epoch
        self.iou_multiplier = iou_multiplier
        self.obj_multiplier = obj_multiplier
        self.prepare_weights()
        self.indices = np.random.choice(np.arange(len(self.dataset)), size=self.iters_per_epoch, p=self.weights)
    
    def prepare_weights(self):
        WEIFHT_CACHE_PATH = "data/weight_cache.pkl"
        cache = {}
        key = f"iou={self.iou_multiplier}, obj={self.obj_multiplier}"
        if os.path.exists(WEIFHT_CACHE_PATH):
            with open(WEIFHT_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            if key in cache.keys() and len(cache[key]) == len(self):
                self.weights = cache[key]
                print(f"loaded weights from cache: {key}")
                return
        self.weights = []
        pbar = tqdm(range(len(self.dataset)))
        pbar.set_description(f"calculating weights for dataset")
        for i in range(len(self.dataset)):
            pbar.update(1)
            data = self.dataset[i]
            if isinstance(data, list):
                data = data[0]
            anns = data["ann_info"]
            num_obj = len(anns["bboxes"])
            bboxes = torch.from_numpy(anns["bboxes"])
            ious = bbox_overlaps(bboxes, bboxes, mode='iou')
            ious[torch.arange(len(ious)), torch.arange(len(ious))] = 0
            if len(ious) != 0:
                max_iou = ious.max()
            else:
                max_iou = 0
            self.weights.append(self.calc_weight(max_iou, num_obj))
        self.weights = np.array(self.weights)
        self.weights /= self.weights.sum()
        cache[key] = self.weights
        with open(WEIFHT_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    
    def calc_weight(self, max_iou, num_obj):
        iou_weight = sum(pow(max_iou, p) * m for (p, m) in self.iou_multiplier)
        obj_weight = sum(pow(num_obj, p) * m for (p, m) in self.obj_multiplier)
        return 1 + iou_weight + obj_weight
            
    def __len__(self):
        return self.iters_per_epoch
    
    def __getitem__(self, idx):
        if idx == 0:
            self.indices = np.random.choice(np.arange(len(self.dataset)), size=self.iters_per_epoch, p=self.weights)
        return self.dataset[self.indices[idx]]
    
    def __iter__(self):
        indices = np.random.choice(np.arange(len(self.dataset)), size=self.iters_per_epoch, p=self.weights)
        for idx in indices:
            yield self.dataset[idx]