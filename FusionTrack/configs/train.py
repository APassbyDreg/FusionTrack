stages = [
    {
        "name": "finetune-detector-mot",
        "load_pretrained": True,
        "epoch": 4,
        "bs": 4,
        "lr": 1e-5,
        "datasets": [
            "mot15",
            "mot16",
            "mot17",
            "mot20",
            "crowdhuman"
        ],
        "dataset_type": "img"
    },
    {
        "name": "pretrain",
        "load_pretrained": True,
        "epoch": 4,
        "bs": 4,
        "lr": 2e-5,
        "datasets": [
            "dancetrack",
            "mot17",
            "mot20",
            "crowdhuman"
        ],
        "dataset_type": "img"
    },
    {
        "name": "train-all",
        "epoch": 2,
        "bs": 1,
        "lr": 2e-5,
        "datasets": [
            "dancetrack",
            "mot17",
            "mot20",
            "crowdhuman"
        ],
        "dataset_type": "vid",
        "clip_length": 4
    },
    {
        "name": "train-all-weighted",
        "epoch": 6,
        "bs": 1,
        "lr": 2e-5,
        "datasets": [
            "dancetrack",
            "dancetrack_val",
            "mot15",
            "mot16",
            "mot17",
            "mot20",
            "crowdhuman"
        ],
        "use_weighted_dataset": True,
        "dataset_type": "vid",
        "clip_length": 3
    },
    {
        "name": "finetune-dancetrack", # optional
        "epoch": 2,
        "bs": 1,
        "lr": 1e-5,
        "datasets": [
            "dancetrack",
            "dancetrack_val"
        ],
        "dataset_type": "vid",
        "clip_length": 4
    },
    {
        "name": "finetune-mot-all",
        "epoch": 2,
        "bs": 1,
        "lr": 1e-5,
        "datasets": [
            "mot15",
            "mot16",
            "mot17",
            "mot20",
            "crowdhuman"
        ],
        "dataset_type": "vid",
        "clip_length": 4
    },
    *list(
        {
            "name": f"finetune-mot-{year}",
            "epoch": 6,
            "bs": 1,
            "lr": 2e-5,
            "datasets": [
                f"mot{year}",
            ],
            "dataset_type": "vid",
            "clip_length": 4
        } for year in [15, 16, 17, 20]
    ),
    {
        "name": "test",
        "epoch": 1,
        "bs": 1,
        "lr": 1e-4,
        "datasets": [
            "dancetrack"
        ],
        "dataset_type": "vid",
        "clip_length": 2
    },
]


def get_train_cfg(id_or_name):
    try:
        idx = int(id_or_name)
        return stages[idx]
    except ValueError:
        for stage in stages:
            if stage["name"] == id_or_name:
                return stage
        raise ValueError(f"Unknown train stage: {id_or_name}")