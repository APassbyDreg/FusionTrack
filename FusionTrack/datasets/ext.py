from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadInstanceID:
    def __init__(self) -> None:
        pass
    def __call__(self, results):
        results['gt_instance_ids'] = results['ann_info']['instance_ids'].copy()
        return results