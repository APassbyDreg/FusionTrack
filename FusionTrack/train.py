# ---------------------------------- basics ---------------------------------- #

import importlib
import torch
import numpy as np
import argparse
import copy
import time
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Train a tracker')
    parser.add_argument('-s', '--stage', type=str, default="")
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-v', '--vmodel', type=int, default=3)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-j', '--joint_stage', type=str, default="")
    parser.add_argument('--expname', type=str, default="raw")
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--wandb', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0)
    parser.add_argument('--bs', type=int, default=0)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--init_track_branch', action='store_true', default=0)
    parser.add_argument('--disable-lr-decay', action='store_true', default=0)
    return parser.parse_args()

args = parse_args()
train_cfg = importlib.import_module('configs.train').get_train_cfg(args.stage)
train_cfg.update({
    "stage": args.stage,
    "device": args.device,
    "ckpt": args.ckpt,
    "seed": args.seed,
    "expname": args.expname,
})
if args.epoch > 0:
    train_cfg["epoch"] = args.epoch
if args.lr > 0:
    train_cfg["lr"] = args.lr
if args.bs > 0:
    train_cfg["bs"] = args.bs
    
joint_training = args.joint_stage != ""
if joint_training:
    joint_train_cfg = importlib.import_module('configs.train').get_train_cfg(args.joint_stage)
    train_cfg["joint_cfg"] = joint_train_cfg
    
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device(f"cuda:{args.device}")
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S') 

if args.delay > 0:
    print(f"[{timestamp}] the training will start after {args.delay} seconds..")
    time.sleep(args.delay)   

# ---------------------------------- dataset --------------------------------- #

from datasets.loader import BatchedPretrainLoader, BatchedVideoLoader
from datasets.datasets import JointDataset, PseudoVideoDataset, WeightedDataset
from mmtrack.datasets import build_dataset

use_weighted_dataset = train_cfg.get("use_weighted_dataset", False)
dataset_cfgs = importlib.import_module('configs.dataset')

def make_single_dataset(base_cfg, parent_cfg):
    base_cfg = copy.deepcopy(base_cfg)
    if parent_cfg["dataset_type"] == "vid" and base_cfg["type"] != "CocoDataset":
        base_cfg["ref_img_sampler"] = dataset_cfgs.make_clip_sampler(parent_cfg["clip_length"])
    else:
        pass
    return build_dataset(base_cfg)

def make_full_dataset(cfg):
    datasets = []
    cfg = copy.deepcopy(cfg)
    if "dancetrack" in cfg["datasets"]:
        datasets.append(make_single_dataset(dataset_cfgs.dance_track["train"], cfg))
    if "dancetrack_val" in cfg["datasets"]:
        datasets.append(make_single_dataset(dataset_cfgs.dance_track["val"], cfg))
    if "mot17" in cfg["datasets"]:
        datasets.append(make_single_dataset(dataset_cfgs.mot_train[15], cfg))
    if "mot17" in cfg["datasets"]:
        datasets.append(make_single_dataset(dataset_cfgs.mot_train[16], cfg))
    if "mot17" in cfg["datasets"]:
        datasets.append(make_single_dataset(dataset_cfgs.mot_train[17], cfg))
    if "mot20" in cfg["datasets"]:
        datasets.append(make_single_dataset(dataset_cfgs.mot_train[20], cfg))
    if "crowdhuman" in cfg["datasets"]:
        if cfg["dataset_type"] == "img":
            datasets.append(make_single_dataset(dataset_cfgs.crowdhuman_pretrain, cfg))
        elif cfg["dataset_type"] == "vid":
            datasets.append(
                PseudoVideoDataset(
                    make_single_dataset(dataset_cfgs.crowdhuman_pretrain, cfg),
                    dataset_cfgs.make_clip_sampler(cfg["clip_length"])
                )
            )
        else:
            raise NotImplementedError(f"Unknown dataset type: {train_cfg['dataset_type']}")
    dataset = JointDataset(datasets)
    return dataset

dataset = make_full_dataset(train_cfg)
if joint_training:
    joint_training_dataset = make_full_dataset(joint_train_cfg)

if use_weighted_dataset:
    dataset = WeightedDataset(dataset)

def make_dataloader(cfg, dataset):
    if cfg["dataset_type"] == "vid":
        data_loader = BatchedVideoLoader(
            dataset, 
            batch_size=cfg["bs"],
            num_workers=cfg["bs"],
            shuffle=True, 
            img_pipeline=dataset_cfgs.train_vid_pipeline
        )
    elif cfg["dataset_type"] == "img":
        data_loader = BatchedPretrainLoader(
            dataset, 
            batch_size=cfg["bs"],
            num_workers=cfg["bs"],
            shuffle=True, 
            img_pipeline=dataset_cfgs.train_img_pipeline,
            repeat=cfg.get("repeat", 1)
        )
    else:
        raise NotImplementedError(f"Unknown dataset type: {train_cfg['dataset_type']}")
    data_loader.to(device)
    return data_loader

data_loader = make_dataloader(train_cfg, dataset)
if joint_training:
    joint_training_data_loader = make_dataloader(joint_train_cfg, joint_training_dataset)

# ----------------------------------- model ---------------------------------- #

from models import TransMotion

if args.vmodel == 2:
    from models.v2 import get_detector
elif args.vmodel == 3:
    from models.v3 import get_detector
elif args.vmodel == 4:
    from models.v4 import get_detector
else:
    raise NotImplementedError(f"Unknown model version: {args.vmodel}")

ckpt_path = args.ckpt

model_cfgs = importlib.import_module('configs.model')
model_cfg = model_cfgs.get_model_cfg(args.expname)
model_cfg["tracker_cfg"]["num_parallel_batches"] = train_cfg["bs"]
det = get_detector(pretrained=train_cfg.get("load_pretrained", False))
tracker = TransMotion(detector=det, **model_cfg)
tracker.to(device)

if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location=device)
    tracker.load_state_dict(ckpt['state_dict'], strict=False)

if train_cfg.get("freeze_pretrained", False):
    tracker.freeze_pretrained()

if train_cfg.get("freeze_not_decoder_head", False):
    tracker.freeze_not_decoder_head()
            
if args.init_track_branch:
    tracker.detector.init_track_branch_with_det_branch()

# ---------------------------------- logging --------------------------------- #

import os

run_dir = os.path.join(f"runs/v{args.vmodel}_{train_cfg['name']}", timestamp)
os.makedirs(run_dir, exist_ok=True)

import wandb
import json

n_epoch = train_cfg["epoch"]
n_iters = len(data_loader) + (len(joint_training_data_loader) if joint_training else 0)
log_interval = 10
ckpt_interval = 1000
keep_ckpts = 2
ckpts = []

use_wandb = args.wandb is not None

config = dict(
    model=model_cfg,
    train=train_cfg,
)

if use_wandb:
    wandb.init(
        entity="martinzhe",
        project="TransMotion",
        name=args.wandb,
        save_code=True,
        group="fine-tune",
        config=config
    )

log_file = open(os.path.join(run_dir, "log.txt"), "w")

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f)

def log_to_file(states):
    outs = {}
    for k, v in states.items():
        if isinstance(v, torch.Tensor):
            outs[k] = float(v)
        else:
            outs[k] = v
    log_file.write(json.dumps(outs) + "\n")

def log(states):
    avg_time_per_step = states['total_time'] / states['total_step']
    epoch = states['epoch']
    iter = states['iter']
    eta = ((n_epoch - epoch + 1) * n_iters - iter) * avg_time_per_step
    eta = datetime.timedelta(seconds=eta)
    print(f"[{states['timestamp']}] epoch: {states['epoch']:4d}, iter: {states['iter']:4d}/{n_iters}, loss: {float(states['loss']):.6f}, lr: {states['lr']:4e}, grad_norm: {states['grad_norm']:4f}, time: {states['time']:.4f}, data: {states['data_time']:.4f}, eta: {eta}")
    log_to_file(states)
    if use_wandb:
        wandb.log(states, step=states['total_step'])

# -------------------------------- optimizing -------------------------------- #

from torch import optim, nn

optimizer = optim.AdamW(tracker.parameters(), lr=train_cfg["lr"], weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch * n_iters) if not args.disable_lr_decay else None

def merge_losses(losses):
    loss = 0
    for k, v in losses.items():
        if "loss" in k and v.requires_grad:
            if len(v.shape) == 0:
                loss += v
            elif len(v.shape) == 1:
                loss += v[0]
    return loss

def clip_grad(m : nn.Module, clip_value=1e2):
    if hasattr(m, "parameters"):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, m.parameters(True)))
    else:
        params = [m]
    return nn.utils.clip_grad.clip_grad_norm_(params, clip_value)

# ---------------------------- main training loop ---------------------------- #

def save_ckpt(path, epoch=None, iter=None):
    torch.save({
        "state_dict": tracker.state_dict(),
        "optimizer": optimizer.state_dict(),
        "meta": {
            "epoch": epoch,
            "iter": iter
        }
    }, path)
    
def train():
    running_time = None
    total_step = 0
    t_start = time.time()
    accumulated_loss = 0
    t0 = t1 = None
    t2 = time.time()
    for e in range(n_epoch):
        main_iter = iter(data_loader)
        main_remain = len(data_loader)
        joint_iter = iter(joint_training_data_loader) if joint_training else None
        joint_remain = len(joint_training_data_loader) if joint_training else 0
        for i in range(n_iters):
            p = np.random.rand()
            if p > (main_remain / (main_remain + joint_remain)):
                data = next(joint_iter)
                joint_remain -= 1
            else:
                data = next(main_iter)
                main_remain -= 1
            # preprocess
            t0 = time.time()
            data_time = t0 - t2
            # train
            optimizer.zero_grad()
            states, _ = tracker.forward_train(**data)
            loss = merge_losses(states)
            if isinstance(loss, torch.Tensor):
                loss.backward()
                grad_norm = clip_grad(tracker, clip_value=1e2)
            else:
                grad_norm = None
            optimizer.step()
            if not args.disable_lr_decay:
                scheduler.step()
            # record
            t1 = time.time()
            total_step += 1
            running_time = t1 - t0 if running_time is None else running_time * 0.95 + (t1 - t0) * 0.05
            total_time = t1 - t_start
            accumulated_loss += float(loss)
            # log
            if i % log_interval == 0 and i != 0:
                timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                states.update({
                    "timestamp": timestamp, 
                    "loss": accumulated_loss / log_interval, # avg loss in interval
                    "lr": optimizer.param_groups[0]["lr"], 
                    "time": running_time,
                    "data_time": data_time,
                    "total_time": total_time,
                    "total_step": total_step,
                    "grad_norm": float(grad_norm),
                    "epoch": e + 1,
                    "iter": i,
                })
                states = {k.replace(".", "/"): v for k, v in states.items()}
                log(states)
                accumulated_loss = 0
            if total_step % ckpt_interval == 0:
                ckpt_path = os.path.join(run_dir, f"{total_step}step.pth")
                ckpts.append(ckpt_path)
                while len(ckpts) > keep_ckpts:
                    rm_path = ckpts.pop(0)
                    os.remove(rm_path)
                save_ckpt(ckpt_path, e, i)
            # record end time
            t2 = time.time()
    save_ckpt(os.path.join(run_dir, "final.pth"))
    for ckpt_path in ckpts:
        os.remove(ckpt_path)

import traceback

try:
    train()
except Exception as e:
    with open("train_err.log", "w") as f:
        traceback.print_exc(file=f)
        f.write(str(e))
        f.write(str(e.__traceback__))
    raise e