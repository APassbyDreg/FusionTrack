import importlib
import os
import time
import argparse
import json
import copy
import shutil
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from PIL import Image

from models import TransMotion
from datasets.loader import EvalLoader
from mmtrack.datasets import build_dataset


# ------------------------------- cmd arguments ------------------------------ #
parser = argparse.ArgumentParser(description='Train a tracker')
parser.add_argument('-ds', '--dataset', type=str, choices=["dancetrack_val", "dancetrack_test", "mot17_test", "mot17_train", "mot17_val"], default="dancetrack_test")
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-v', '--vmodel', type=int)
parser.add_argument('--ckpt', type=str)
parser.add_argument('--expname', type=str, default="raw")
parser.add_argument('--vis', action="store_true")
parser.add_argument('--val', action="store_true")
args = parser.parse_args()

# ---------------------------------- basics ---------------------------------- #
torch.backends.cudnn.deterministic = True
device = torch.device(f"cuda:{args.device}")


# ---------------------------------- dataset --------------------------------- #
dataset_cfgs = importlib.import_module('configs.dataset')
dataset_cfgs = importlib.reload(dataset_cfgs)
ds_cfg = None
assert "val" in args.dataset or not args.val
if args.dataset == "dancetrack_val":
    ds_cfg = dataset_cfgs.dancetrack_inference["val"]
elif args.dataset == "dancetrack_test":
    ds_cfg = dataset_cfgs.dancetrack_inference["test"]
elif args.dataset == "mot17_train":
    ds_cfg = dataset_cfgs.mot_inference["train"][17]
elif args.dataset == "mot17_test":
    ds_cfg = dataset_cfgs.mot_inference["test"][17]
elif args.dataset == "mot17_val":
    ds_cfg = dataset_cfgs.mot_inference["half-val"][17]
else:
    raise ValueError(f"unsupport dataset: {args.dataset}")
dataset = build_dataset(copy.deepcopy(ds_cfg))
data_loader = EvalLoader(dataset, device=device)
data_loader.to(device)


# ----------------------------------- model ---------------------------------- #
if args.vmodel == 2:
    from models.v2 import get_detector
elif args.vmodel == 3:
    from models.v3 import get_detector
elif args.vmodel == 4:
    from models.v4 import get_detector
else:
    raise NotImplementedError(f"Unknown model version: {args.vmodel}")

model_cfgs = importlib.import_module('configs.model')
model_cfg = model_cfgs.get_model_cfg(args.expname)
model_cfg["tracker_cfg"]["num_parallel_batches"] = 1

det = get_detector()
tracker = TransMotion(detector=det, **model_cfg)
tracker.to(device)

ckpt = torch.load(args.ckpt, map_location=device)
tracker.load_state_dict(ckpt["state_dict"], strict=False)
tracker.eval()


# --------------------------- run model on test set -------------------------- #
print("=" * 100)
print(f"Evaluating {args.dataset} dataset")
print("=" * 100)
results = []
cnt = 0
t0 = time.time()
for data in tqdm(data_loader):
    with torch.no_grad():
        result = tracker.simple_test(**data)
        results.append(result)
    cnt += 1
t1 = time.time()
dt = t1 - t0
    

# ---------------------------------- output ---------------------------------- #
timestamp = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
out_dir = f"results/{args.dataset}/{timestamp}"
os.makedirs(out_dir, exist_ok=True)

det_bboxes = [r["det_bboxes"] for r in results]
track_bboxes = [r["track_bboxes"] for r in results]
results_dict = {
    "det_bboxes": det_bboxes,
    "track_bboxes": track_bboxes,
}
dataset.format_results(results_dict, out_dir)

# rename and make archive for dancetrack
if args.dataset == "dancetrack_test":
    shutil.move(f"{out_dir}/track", f"{out_dir}/tracker")
    shutil.make_archive(out_dir, "zip", f"{out_dir}")

with open(f"{out_dir}/info.json", "w") as f:
    info = {
        "model": f"v{args.vmodel}",
        "expname": args.expname,
        "ckpt": args.ckpt,
        "time": dt,
        "timestamp": timestamp
    }
    if args.val:
        metrics = dataset.evaluate(results_dict)
        info["metrics"] = metrics
    json.dump(info, f)


# ---------------------------- visualize to videos --------------------------- #
colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "white"]

def plot_bbox(ax, bbox, idx, conf=None):
    title = f"{idx}" if conf is None else f"{idx}: {conf:.4f}"
    color = colors[idx % len(colors)]
    ax.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]], [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]], color=color)
    ax.text(bbox[0], bbox[1] - 10, title, color=color)
    
if args.vis:
    ds_cfg["pipeline"] = []
    ds = build_dataset(copy.deepcopy(ds_cfg))
    out_dir = f"val_videos/{args.dataset}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    current_video_id = None
    out = None
    pbar = tqdm(dataset.data_infos)
    for i, data in enumerate(pbar):
        vid = data["file_name"].split("/")[0]
        if vid != current_video_id:
            if out is not None:
                out.release()
            path = f"{out_dir}/{vid}.mp4"
            if os.path.exists(path):
                os.remove(path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 10, (data["width"], data["height"]))
            current_video_id = vid
        plt.gcf().clear()
        impath = os.path.join(ds_cfg["img_prefix"], data["filename"])
        img = Image.open(impath)
        plt.imshow(img)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.title(f"frame {data['frame_id']} / sample {i}")
        plt.axis("off")
        for box in results_dict["track_bboxes"][i][0]:
            plot_bbox(plt.gca(), box[1:5], int(box[0]))
        plt.gcf().canvas.draw()
        frame = cv2.cvtColor(np.asarray(plt.gcf().canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        frame = cv2.resize(frame, (data["width"], data["height"]), interpolation = cv2.INTER_CUBIC)
        out.write(frame)
        pbar.set_description(f"frame {data['frame_id']} of video {vid}")
    out.release()
    shutil.make_archive(out_dir, 'zip', out_dir)