import numpy as np
import torch

import os.path as osp

from mmdet.datasets.pipelines import Compose

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from .transforms import make_transforms

from mmdet.core.bbox import bbox_cxcywh_to_xyxy

def to_target_device(frame, device):
    keys = ["img", "gt_bboxes", "gt_labels", "gt_instance_ids"]
    for k in keys:
        frame[k] = frame[k].to(device)
    return frame
            

class BatchedVideoLoader:
    def __init__(
        self, 
        dataset, 
        batch_size=1,
        num_workers=16,
        shuffle=True,
        img_pipeline=None,
        device="cpu",
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.shuffle = shuffle
        self.pipeline = Compose(img_pipeline)
        self.n = len(dataset)
        self.num_batches = self.n // self.batch_size
        self.aug = make_transforms('train')
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        self.thread_pool_loader = ThreadPoolExecutor(max_workers=16)
    
    def to(self, device):
        self.device = device
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(self.num_batches):
            # [batch, frame_cnt]
            batches = [self.dataset[indices[i * self.batch_size + j]] for j in range(self.batch_size)]
            # get data
            data = self.aug_batched_videos(batches)
            yield { "frames": data }
    
    def aug_video(self, data):
        imgs = list(self.thread_pool_loader.map(
            lambda f: Image.open(osp.join(f["img_prefix"], f["img_info"]["filename"])), 
            data
        ))
        targets = [
            dict(
                obj_ids = torch.as_tensor(f["ann_info"]["instance_ids"]),
                labels = torch.as_tensor(f["ann_info"]["labels"]),
                boxes = torch.as_tensor(f["ann_info"]["bboxes"]),
                area = torch.as_tensor(
                    (f["ann_info"]["bboxes"][:, 3] - f["ann_info"]["bboxes"][:, 1]) * \
                    (f["ann_info"]["bboxes"][:, 2] - f["ann_info"]["bboxes"][:, 0])
                ),
            ) for f in data
        ]
        imgs, targets = self.aug(imgs, targets)
        return imgs, targets
        
    def aug_batched_videos(self, batched_vid):
        outs = list(self.thread_pool.map(self.aug_video, batched_vid))
        data = []
        for f in range(len(batched_vid[0])):
            imgs = [outs[b][0][f] for b in range(self.batch_size)]
            targets = [outs[b][1][f] for b in range(self.batch_size)]
            img_data = torch.stack(imgs, dim=0).to(self.device)         # TODO: handel batchsize > 1
            box_data = [t["boxes"].to(self.device) for t in targets]
            cls_data = [t["labels"].to(self.device) for t in targets]
            instance_data = [t["obj_ids"].to(self.device) for t in targets]
            meta_data = []
            for b in range(self.batch_size):
                # to xyxy in pixel space to align with MMDetection
                box_data[b] = bbox_cxcywh_to_xyxy(box_data[b])
                box_data[b] = box_data[b] * targets[b]["size"][[1, 0, 1, 0]].to(self.device)
                meta = {}
                meta["filename"] = osp.join(batched_vid[b][f]["img_prefix"], batched_vid[b][f]["img_info"]["filename"])
                meta["frame_id"] = batched_vid[b][f]["img_info"]["frame_id"]
                meta["img_shape"] = (img_data.shape[2], img_data.shape[3], img_data.shape[1])
                meta["ori_shape"] = (batched_vid[b][f]["img_info"]["height"], batched_vid[b][f]["img_info"]["width"], 3)
                meta_data.append(meta)
            data.append(
                dict(
                    img=img_data,
                    img_metas=meta_data,
                    gt_bboxes=box_data,
                    gt_labels=cls_data,
                    gt_instance_ids=instance_data,
                )
            )
        return data
    

class BatchedPretrainLoader:
    def __init__(
        self, 
        dataset, 
        batch_size=1,
        num_workers=16,
        shuffle=True,
        img_pipeline=None,
        repeat=1,
        device="cpu",    
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.shuffle = shuffle
        self.pipeline = Compose(img_pipeline)
        self.n = len(dataset)
        self.repeat = repeat
        self.num_batches = self.n // self.batch_size
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        
    def to(self, device):
        self.device = device
    
    def __len__(self):
        return self.num_batches * self.repeat
    
    def __iter__(self):
        indices = np.arange(self.n)
        for r in range(self.repeat):
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(self.num_batches):
                batched_data = [self.dataset[indices[i * self.batch_size + j]] for j in range(self.batch_size)]
                # convert
                batched_data = list(self.thread_pool.map(self.pipeline, batched_data))
                img_data = torch.stack([d["img"] for d in batched_data], dim=0).to(self.device)
                meta_data = [d["img_metas"].data for d in batched_data]
                box_data = [d["gt_bboxes"].to(self.device) for d in batched_data]
                cls_data = [d["gt_labels"].to(self.device) for d in batched_data]
                instance_data = [d["gt_instance_ids"].to(self.device) for d in batched_data]
                yield dict(
                    img=img_data,
                    img_metas=meta_data,
                    gt_bboxes=box_data,
                    gt_labels=cls_data,
                    gt_instance_ids=instance_data,
                )
            
            
class EvalLoader:
    def __init__(self, dataset, device="cpu") -> None:
        self.dataset = dataset
        self.n = len(dataset)
        self.device = device
        
    def to(self, device):
        self.device = device
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        img_metas = [data["img_metas"].data]
        img = data["img"].to(self.device).unsqueeze(0)
        return dict(img=img, img_metas=img_metas)