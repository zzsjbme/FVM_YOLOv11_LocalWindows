import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info
import matplotlib.pyplot as plt

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = r'runs/experient_1111/GDM_copy_must_2025_-yolov11/weights/best.pt'
    model = YOLO(model_path) # 选择训练好的权重路径
    result = model.val(
                        data=r'datasets4/GDM_copy2/GDM_copy2.yaml',
                        split='test', # split可以选择train、val、test 根据自己的数据集情况来选择.
                        imgsz=640,
                        project=r'runs/val_1111',
                        save_json=True,
                        name='GDM_enhance_yolov11',
                       # project和name参数用于控制验证结果的保存路径和命名规则
                        )
    print(f"验证结果保存路径为: {result.save_dir}")
    if model.task == 'detect': # 仅目标检测任务适用
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)
        print('-'*20 + '论文上的数据以以下结果为准' + '-'*20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图", "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)", "Model File Size"]
        # 参数数量转换为百万(M)为单位
        params_millions = n_p / 1e6  # 转换为百万
        model_info_table.add_row([
            f'{flops:.2f}',
            f'{params_millions:,.2f}M',  # 修正后的格式
            f'{preprocess_time_per_image / 1000:.6f}s',
            f'{inference_time_per_image / 1000:.6f}s',
            f'{postprocess_time_per_image / 1000:.6f}s',
            f'{1000 / all_time_per_image:.2f}',
            f'{1000 / inference_time_per_image:.2f}',
            f'{get_weight_size(model_path)}MB'
        ])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                                        model_names[idx],
                                        f"{result.box.p[idx]:.4f}",
                                        f"{result.box.r[idx]:.4f}",
                                        f"{result.box.f1[idx]:.4f}",
                                        f"{result.box.ap50[idx]:.4f}",
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        model_metrice_table.add_row([
                                    "all(平均数据)",
                                    f"{result.results_dict['metrics/precision(B)']:.4f}",
                                    f"{result.results_dict['metrics/recall(B)']:.4f}",
                                    f"{np.mean(result.box.f1[:length]):.4f}",
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}",
                                    f"{np.mean(result.box.all_ap[:length, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))

        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)
        print('-'*20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-'*20)