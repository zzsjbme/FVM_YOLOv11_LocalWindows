import os
from pathlib import Path

# 创建目录和空文件
font_dir = Path('/root/.config/Ultralytics/')
font_dir.mkdir(parents=True, exist_ok=True)
font_file = font_dir / 'Arial.Unicode.ttf'

# 创建空文件避免下载
if not font_file.exists():
    font_file.touch()
from ultralytics import YOLO


def train_model():
    # 定义要训练的模型配置列表
    models_to_train = [
        # {
        #     'config_path': r'ultralytics/cfg/models/11/yolo11-LocalWindowAttention-medium.yaml',
        #     'name_suffix': 'LocalWindowAttention-medium'
        # },
         {
            'config_path': r'ultralytics/cfg/models/11/yolo11.yaml',
            'name_suffix': 'yolov11'
        },
        # {
        #     'config_path': r'ultralytics/cfg/models/11/yolo11-LocalWindowAttention-P4.yaml',
        #     'name_suffix': 'LocalWindowAttention-P4'
        # },
        # {
        #     'config_path': r'ultralytics/cfg/models/11/yolo11-LocalWindowAttention-P5.yaml',
        #     'name_suffix': 'LocalWindowAttention-P5'
        # },
        # {
        #     'config_path': r'ultralytics/cfg/models/v3/yolov3-tiny.yaml',
        #     'name_suffix': 'yolov3-tiny'
        # },
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
            name=f"GDM_enhance_yolov5",
            pretrained=False,
            amp=False
        )


# 主模块保护块（Windows 多进程必须）
if __name__ == '__main__':
    train_model()