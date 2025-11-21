import os
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

# MONAI 导入
from monai.utils import set_determinism
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    OneOf,  # <--- 核心：用于实现“必然二选一”
    LoadImage,
    SaveImage,
    EnsureChannelFirst,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandGaussianSharpend,
    MapTransform,
    Randomizable,
    CastToTyped
)


# -------------------------------------------------------
# 1. 核心：原子化翻转与坐标同步类
# -------------------------------------------------------
class RandYOLOFlipd(Randomizable, MapTransform):
    """
    自定义变换：
    如果被调用（且概率满足），则同时翻转图像和 YOLO 坐标。
    """

    def __init__(self, keys, prob=1.0, spatial_axis=1):
        """
        spatial_axis: 0=垂直翻转(y轴), 1=水平翻转(x轴)
        """
        super().__init__(keys)
        self.prob = prob
        self.spatial_axis = spatial_axis
        self._do_transform = False

    def randomize(self, data=None):
        # 使用 MONAI 的随机生成器
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        self.randomize(data)

        if not self._do_transform:
            return data

        d = dict(data)

        # --- A. 图像翻转 ---
        # 图像 [C, H, W]。 dim=1是H(垂直), dim=2是W(水平)
        dim = self.spatial_axis + 1
        d[self.keys[0]] = torch.flip(d[self.keys[0]], dims=[dim])

        # --- B. 坐标同步更新 (核心逻辑) ---
        # 只要图像翻了，坐标必须翻，确保原子性
        ann_key = self.keys[1]
        if ann_key in d and d[ann_key]:
            old_anns = d[ann_key]
            new_anns = []

            for ann in old_anns:
                # YOLO格式: class, x_center, y_center, w, h
                cls_id, x_c, y_c, w, h = ann

                if self.spatial_axis == 1:  # 水平翻转 -> x 变
                    new_x_c = 1.0 - x_c
                    new_anns.append((cls_id, new_x_c, y_c, w, h))

                elif self.spatial_axis == 0:  # 垂直翻转 -> y 变
                    new_y_c = 1.0 - y_c
                    new_anns.append((cls_id, x_c, new_y_c, w, h))

            d[ann_key] = new_anns

        return d



class MedicalDataAugmentor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义翻转操作
        # prob=1.0 意味着只要选中这个操作，就一定执行
        flip_horizontal = RandYOLOFlipd(keys=["image", "annotations"], prob=1.0, spatial_axis=1)
        flip_vertical = RandYOLOFlipd(keys=["image", "annotations"], prob=1.0, spatial_axis=0)

        self.transforms = Compose([

            OneOf(
                transforms=[flip_horizontal, flip_vertical]
            ),


            RandGaussianSharpend(
                keys=["image"],
                sigma1_x=(0.5, 1.0), sigma1_y=(0.5, 1.0),
                alpha=(10.0, 30.0),
                prob=1.0
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                prob=1.0
            ),
            RandAdjustContrastd(
                keys=["image"],
                gamma=(0.7, 1.3),
                prob=1.0
            ),

            # 格式转换
            CastToTyped(keys=["image"], dtype=np.float32)
        ])

    def dataAugment(self, img, annotations):
        data_dict = {
            "image": img,
            "annotations": deepcopy(annotations)
        }
        aug_data = self.transforms(data_dict)
        return aug_data["image"], aug_data["annotations"]


class MedicalImageDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(img_dir)
                            if f.lower().endswith(('jpg', 'png', 'jpeg', 'bmp'))]

        self.loader = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.img_dir, img_name)

        img = self.loader(img_path)

        txt_path = os.path.join(self.label_dir, f"{base_name}.txt")
        annotations = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        annotations.append((cls_id, *coords))

        return {
            'image': img,
            'annotations': annotations,
            'base_name': base_name
        }


class MONAIDataSaver:
    def __init__(self, image_output_dir, label_output_dir):
        self.image_output_dir = image_output_dir
        self.label_output_dir = label_output_dir
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.label_output_dir, exist_ok=True)

    def save_data(self, img: MetaTensor, annotations, base_name, suffix):
        # 安全转换：保证数据在 0-255 且为 uint8
        img_to_save = torch.clip(img, 0, 255).type(torch.uint8)

        # 更新 Meta 数据以确保文件名正确
        if isinstance(img_to_save, MetaTensor):
            fake_path = os.path.join(self.image_output_dir, f"{base_name}.jpg")
            img_to_save.meta["filename_or_obj"] = fake_path

        saver = SaveImage(
            output_dir=self.image_output_dir,
            output_postfix=suffix,
            output_ext=".jpg",
            resample=False,
            scale=None,
            separate_folder=False,
            print_log=False
        )

        saver(img_to_save)

        # 保存标签
        txt_filename = f"{base_name}_{suffix}.txt"
        txt_path = os.path.join(self.label_output_dir, txt_filename)

        if annotations:
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")


def custom_collate_fn(batch):
    return {
        'image': [item['image'] for item in batch],
        'annotations': [item['annotations'] for item in batch],
        'base_name': [item['base_name'] for item in batch]
    }


# -------------------------------------------------------
# 5. 主程序
# -------------------------------------------------------
def main():
    # 设置随机种子，确保可复现
    set_determinism(seed=42)

    IMAGE_DIR = r"datasets4/GDM/images/train"
    LABEL_DIR = r"datasets4/GDM/labels/train"
    IMAGE_OUTPUT_DIR = r"datasets4/GDM_enhance_must/images/train"
    LABEL_OUTPUT_DIR = r"datasets4/GDM_enhance_must/labels/train"
    AUG_PER_IMAGE = 1

    if not os.path.exists(IMAGE_DIR):
        print("错误：目录不存在")
        return

    dataset = MedicalImageDataset(IMAGE_DIR, LABEL_DIR)
    augmentor = MedicalDataAugmentor()
    saver = MONAIDataSaver(IMAGE_OUTPUT_DIR, LABEL_OUTPUT_DIR)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    total_saved = 0
    print(f"开始处理，总数: {len(dataset)}")

    for batch in dataloader:
        for i in range(len(batch['image'])):
            orig_img = batch['image'][i]
            orig_anns = batch['annotations'][i]
            base_name = batch['base_name'][i]

            if not orig_anns:
                continue

            # 保存原始
            try:
                saver.save_data(orig_img, orig_anns, base_name, "orig")
                total_saved += 1
            except Exception as e:
                print(f"Save orig fail: {e}")

            # 保存增强
            for k in range(AUG_PER_IMAGE):
                try:
                    aug_img, aug_anns = augmentor.dataAugment(orig_img, orig_anns)
                    saver.save_data(aug_img, aug_anns, base_name, f"aug{k}")
                    total_saved += 1
                except Exception as e:
                    print(f"Augment fail {base_name}: {e}")

    print(f"完成。共保存 {total_saved} 张。")


if __name__ == '__main__':
    main()