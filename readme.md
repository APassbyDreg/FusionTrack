# FusionTrack: Multiple Object Tracking with Enhanced Feature Utilization


We introduce FusionTrack to enhance feature utilization within and between frames in MOT using a joint track-detection decoder and a score-guided multi-level query fuser.

## Usage

1. install openmim, mmengine and mmcv
2. install mmdetection and mmtracking in this repository
3. install detrex in this repository
4. install other dependencies
5. prepare datasets in `FusionTrack/data` and generate annotations with mmtracking. The resulting folder structure is as follows
    ```
    FusionTrack
    - data
        - DanceTrack
            - annotations
            - ...
        - MOT
            - MOT15
                - annotations
            - ...
        - CrowdHuman
            - annotations
            - ...
    ```
6. run `FusionTrack/train.py` for training or `FusionTrack/inference.py` for inference

## Performance

| HOTA | DetA | AssA | MOTA | IDF1 |
| ---- | ---- | ---- | ---- | ---- |
| 65.3 | 75.0 | 57.5 | 90.1 | 73.3 |