_base_ = 'rtmdet_tiny_8xb32-300e_coco.py'

model = dict(
    bbox_head=dict(
        num_classes=7
    )
)

# 修改数据集相关配置
data_root = '/home/waas/mmdetection-main/mmdetection-main/datasets/'
metainfo = {
    'classes': (
        '干绒毛血管闭塞',
        '绒毛间质、血管细胞核碎裂',
        '血管壁纤维素沉积',
        '血管扩张',
        '血栓形成',
        '绒毛成熟延迟',
        '无血管绒毛'
    )
}
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-train/annotations/train.json',
        data_prefix=dict(img='VisDrone2019-DET-train/images/')))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-val/annotations/val.json',
        data_prefix=dict(img='VisDrone2019-DET-val/images/')))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='VisDrone2019-DET-test-dev/annotations/test.json',
        data_prefix=dict(img='VisDrone2019-DET-test-dev/images/')))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'VisDrone2019-DET-val/annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'VisDrone2019-DET-test-dev/annotations/test.json')


# 训练循环配置
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环类型：基于轮次（Epoch）的循环（MMEngine标准循环）
    max_epochs=150,  # 最大训练轮次：整个训练过程共运行150个epoch
    val_interval=1)  # 验证间隔：每训练1个epoch后执行一次验证（评估模型当前性能）

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=5), # 检查点保存 - 关键配置
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=-1,  # -1 表示保留所有最佳模型
        save_optimizer=False,  # 为节省空间，可以不保存优化器
    ),
    early_stopping = dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=20,  # 延长容忍周期
        min_delta=0.005,  # 降低最小变化阈值
        rule='greater'
    )
)
load_from='rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

# nohup python tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py > rtmdet-tiny-visdrone.log 2>&1 & tail -f rtmdet-tiny-visdrone.log
# python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py work_dirs/rtmdet_tiny_8xb32-300e_visdrone/epoch_300.pth --show --show-dir test_save
# python tools/test.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py work_dirs/rtmdet_tiny_8xb32-300e_visdrone/epoch_300.pth --tta 
# python tools/analysis_tools/get_flops.py configs/rtmdet/rtmdet_tiny_8xb32-300e_visdrone.py