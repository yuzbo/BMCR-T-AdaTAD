"""Dataset-specific contracts and immutable resource descriptions."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'research/paper'
RECIPE = 'fpw_paper_joint_v2'
SUPPORT_RECIPE = 'support_consistent_paper_v3'
GRAPH_RECIPE = 'graph_tad_v1'
META_KEYS = ['video_name','data_path','fps','duration','snippet_stride','window_start_frame',
             'resize_length','window_size','offset_frames','frame_inds','total_frames']


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if cfg.get('recipe') not in (RECIPE,SUPPORT_RECIPE,GRAPH_RECIPE,'wtr_fasttrack_v1'):
        raise ValueError('Paper recipes have an explicit version and cannot resume v1 experiments')
    if cfg['dataset'] not in ('thumos','anet'):
        raise ValueError(cfg['dataset'])
    if cfg['backbone'] not in ('s','b','internvideo_mq'):
        raise ValueError(cfg['backbone'])
    if cfg['recipe']==GRAPH_RECIPE and (cfg['dataset']!='thumos' or cfg['backbone'] not in ('s','b') or cfg['head']!='point'):
        raise ValueError('The registered graph experiment uses THUMOS VideoMAE-S/B point heads')
    if cfg['recipe']=='wtr_fasttrack_v1':
        if cfg['epochs']!=80 or cfg['eval_epochs']!=[10,20,40,60,80] or cfg.get('frames')!=384:
            raise ValueError('Fast-Track uses one 80-epoch K384 course with inline milestones')
        if cfg.get('graph_kv') or cfg.get('dynamic_budget') or cfg.get('capacity_course'):
            raise ValueError('Base Fast-Track courses keep fixed capacity and ordinary full KV')
    return cfg


def read_resources(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if hasattr(x, 'detach'): return x.detach().cpu().tolist()
        if hasattr(x, 'tolist'): return x.tolist()
        return str(x)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert)+'\n', encoding='utf-8')
    tmp.replace(path)


def build_config(cfg, resources):
    from mmengine.config import Config, ConfigDict
    import h65.full.data  # registers the unchanged THUMOS crop extension
    from . import transforms  # registers resized-video boundary metadata
    family = cfg['backbone']; base = family if family in ('s','b') else 'b'
    dataset = cfg['dataset']; ds = resources['datasets'][dataset]
    name = (f'thumos/e2e_thumos_videomae_{base}_768x1_160_adapter.py' if dataset == 'thumos'
            else f'anet/e2e_anet_videomae_{base}_192x4_160_adapter.py')
    model_cfg = Config.fromfile(str(ROOT/'upstream/configs/adatad'/name))
    if family == 'internvideo_mq':
        iv = Config.fromfile(str(ROOT/'upstream/configs/adatad/ego4d/e2e_ego4d_internvideo_1800x4_192_adapter_lr4e-4.py'))
        model_cfg.model.backbone = copy.deepcopy(iv.model.backbone)
        model_cfg.model.backbone.backbone.total_frames = 768
        model_cfg.model.backbone.backbone.adapter_index = list(range(24))
        model_cfg.model.projection.in_channels = 1024
    model_cfg.model.backbone.custom.pretrain = None
    model_cfg.model.backbone.custom.temporal_checkpointing = False
    model_cfg.model.backbone.backbone.with_cp = False
    model_cfg.model.backbone.backbone.total_frames = 768
    # The paper encoder performs its own 16-frame packing and native pooling.
    for split in ('train','val','test'):
        spec = model_cfg.dataset[split]
        spec.ann_file = ds['annotations']; spec.class_map = ds['class_map']
        spec.data_path = ds['train_videos'] if split == 'train' else ds['test_videos']
        if dataset == 'anet': spec.block_list = ds.get('block_list')
        for tr in spec.pipeline:
            if tr.type == 'LoadFrames' and split == 'train':
                tr.type = 'LoadFramesWithBoundaryValidity' if dataset == 'thumos' else 'PaperResizeFrames'
            if tr.type in ('ConvertToTensor','Collect') and split == 'train':
                if 'gt_boundary_validity' not in tr['keys']: tr['keys'].append('gt_boundary_validity')
            if tr.type == 'Collect': tr.meta_keys = META_KEYS
    model_cfg.evaluation.ground_truth_filename = ds['annotations']
    if dataset == 'anet':
        model_cfg.evaluation.blocked_videos = ds['blocked_annotations']
        model_cfg.post_processing.external_cls.path = ds['classifier']
    model_cfg.paper_point_model = copy.deepcopy(model_cfg.model)
    if cfg['head'] == 'tadtr':
        query = Config.fromfile(str(ROOT/'upstream/configs/_base_/models/tadtr.py')).model
        query.projection.in_channels = model_cfg.model.backbone.backbone.embed_dims
        classes = 20 if dataset == 'thumos' else 1
        query.transformer.num_classes = classes; query.transformer.loss.num_classes = classes
        query.transformer.num_proposals = cfg.get('num_queries', 40)
        model_cfg.model = ConfigDict(dict(**query, backbone=copy.deepcopy(model_cfg.model.backbone)))
        model_cfg.post_processing.pre_nms_topk = 200
    return model_cfg


def dataset_contract(cfg, resources):
    ds = resources['datasets'][cfg['dataset']]
    return dict(dataset=cfg['dataset'], candidate_frames=768,
                detector_length=768 if cfg['dataset']=='thumos' else 192,
                native_length=384, train_ids=ds['train_ids'], test_ids=ds['test_ids'],
                thresholds=[.3,.4,.5,.6,.7] if cfg['dataset']=='thumos' else [.5+i*.05 for i in range(10)])


def dry_description(cfg, resources_path):
    return dict(config=cfg, resources_file=str(resources_path), gpu_execution=False,
                updates='ceil(full_dataset_size/effective_batch)*epochs',
                evaluation='complete target split; inline checkpoints and online terminal',
                criterion='actual total matrix/conv FLOPs and full-test mAP; latency is reported')
