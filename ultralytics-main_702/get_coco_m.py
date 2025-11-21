import warnings
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets

warnings.filterwarnings('ignore')

# 路径配置
ANNO_JSON = r"val_data/test_coco.json"
PRED_JSON = r"runs/val/val_708/GDM_2025_708_yolo11/predictions.json"
RESULT_DIR = r"runs/val/GDM_2025_708_yolo11"


def fix_coco_annotations(anno_path):
    """为边界框添加矩形分割信息"""
    with open(anno_path, 'r') as f:
        data = json.load(f)

    # 为每个标注添加矩形分割
    for ann in data['annotations']:
        if 'segmentation' not in ann or not ann['segmentation']:
            x, y, w, h = ann['bbox']
            # 创建矩形分割多边形 [x1,y1, x2,y1, x2,y2, x1,y2]
            ann['segmentation'] = [[x, y, x + w, y, x + w, y + h, x, y + h]]

    # 创建修复后的文件路径
    fixed_path = os.path.splitext(anno_path)[0] + "_fixed.json"
    with open(fixed_path, 'w') as f:
        json.dump(data, f)

    return fixed_path


if __name__ == '__main__':
    # ==================== 修复标注文件 ====================
    print("修复标注文件中的分割信息...")
    FIXED_ANNO_JSON = fix_coco_annotations(ANNO_JSON)
    print(f"创建修复后的标注文件: {FIXED_ANNO_JSON}")

    # ==================== COCO评估 ====================
    print("执行标准COCO评估...")
    coco_gt = COCO(FIXED_ANNO_JSON)
    coco_pred = coco_gt.loadRes(PRED_JSON)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # ==================== TIDE评估 ====================
    print("执行TIDE详细错误分析...")
    tide = TIDE()

    # 使用修复后的标注文件
    gt_dataset = datasets.COCO(FIXED_ANNO_JSON)
    pred_dataset = datasets.COCOResult(PRED_JSON)

    tide.evaluate_range(gt_dataset, pred_dataset, mode=TIDE.BOX)
    tide.summarize()
    tide.plot(out_dir=RESULT_DIR)
    print(f"TIDE结果已保存至: {RESULT_DIR}")