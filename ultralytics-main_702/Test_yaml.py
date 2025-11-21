# from ultralytics import YOLO
# import torch

# model = YOLO(r'ultralytics/cfg/models/11/yolo11-LocalWindowAttention-P3.yaml', verbose=True)
from ultralytics import YOLO
import os
import warnings
warnings.filterwarnings('ignore')
def train_model():
    # 定义要训练的模型配置列表
    models_to_train = [
        {
            'config_path': r'ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml',
            'name_suffix': 'rtdetr-r18'
        },
    ]

    # 遍历模型配置并依次训练
    for model_config in models_to_train:
        # 创建 YOLO 模型
        model = YOLO(model_config['config_path'])

        # 训练模型，使用对应的名称后缀
        model.train(
            data=r"datasets4/GDM_enhance_must/GDM_enhance_must.yaml",
            batch=16,
            epochs=150,
            patience=20,
            optimizer="SGD",
            project=r"runs/pre_picture_work_710",
            name=f"GDM_enhance_must_2025_rtdetr-r18",
            pretrained=False,
            amp=False
        )


# 主模块保护块（Windows 多进程必须）
if __name__ == '__main__':
    train_model()