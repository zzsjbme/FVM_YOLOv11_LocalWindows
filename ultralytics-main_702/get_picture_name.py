import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image

def convert_yolo_to_voc(yolo_labels_dir, images_dir, output_dir):
    """
    将文件夹下所有YOLO格式标注文件转换为VOC格式XML文件
    
    参数:
    yolo_labels_dir (str): YOLO标注文件所在的文件夹路径
    images_dir (str): 对应的图片文件夹路径（用于获取图片尺寸）
    output_dir (str): 输出VOC XML文件的目录
    """
    # 定义七个类别的名称（按class_id顺序）
    class_names = ["cs", "g", "r", "xgb", "xgk", "xs", "xsxxgb"]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 计数器
    converted_count = 0
    missing_images = 0
    skipped_files = 0
    
    # 遍历YOLO标注文件夹中的所有.txt文件
    for filename in os.listdir(yolo_labels_dir):
        if not filename.lower().endswith('.txt'):
            continue
            
        # 获取不带扩展名的文件名
        base_name = os.path.splitext(filename)[0]
        
        # 查找对应的图片文件
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
            possible_path = os.path.join(images_dir, base_name + ext)
            if os.path.exists(possible_path):
                image_path = possible_path
                break
        
        # 如果没有找到对应的图片文件
        if image_path is None:
            print(f"警告: 找不到 {base_name} 对应的图片文件，跳过此标注")
            missing_images += 1
            continue
        
        try:
            # 打开图片获取尺寸
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"错误: 无法打开图片 {image_path}: {e}")
            skipped_files += 1
            continue
        
        # 完整的YOLO标注文件路径
        yolo_path = os.path.join(yolo_labels_dir, filename)
        
        # 创建XML根元素
        annotation = ET.Element("annotation")
        
        # 添加基本信息
        ET.SubElement(annotation, "folder").text = "VOC2007"
        ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
        
        # 添加图片尺寸信息
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = "3"  # 假设是RGB图片
        
        # 添加来源信息
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "GDM Database"
        
        # 添加分割信息
        ET.SubElement(annotation, "segmented").text = "0"
        
        # 读取YOLO标注
        try:
            with open(yolo_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    data = line.strip().split()
                    
                    # 跳过空行
                    if not data:
                        continue
                    
                    # 验证数据格式
                    if len(data) != 5:
                        print(f"警告: {yolo_path} 第 {line_num} 行格式错误: {line.strip()}，跳过此行")
                        continue
                        
                    try:
                        class_id = int(data[0])
                        center_x = float(data[1])
                        center_y = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
                    except ValueError:
                        print(f"警告: {yolo_path} 第 {line_num} 行包含无效数值: {line.strip()}，跳过此行")
                        continue
                    
                    # 验证类别ID是否有效
                    if class_id < 0 or class_id >= len(class_names):
                        print(f"警告: {yolo_path} 第 {line_num} 行包含无效类别ID {class_id}，跳过此行")
                        continue
                    
                    # 计算边界框坐标 (YOLO格式转换为VOC格式)
                    xmin = max(0, (center_x - width/2) * img_width)
                    ymin = max(0, (center_y - height/2) * img_height)
                    xmax = min(img_width-1, (center_x + width/2) * img_width)
                    ymax = min(img_height-1, (center_y + height/2) * img_height)
                    
                    # 确保边界框有效
                    if xmin >= xmax or ymin >= ymax:
                        print(f"警告: {yolo_path} 第 {line_num} 行无效边界框: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
                        continue
                    
                    # 添加对象
                    obj = ET.SubElement(annotation, "object")
                    ET.SubElement(obj, "name").text = class_names[class_id]
                    ET.SubElement(obj, "pose").text = "Unspecified"
                    ET.SubElement(obj, "truncated").text = "0"
                    ET.SubElement(obj, "difficult").text = "0"
                    
                    bndbox = ET.SubElement(obj, "bndbox")
                    ET.SubElement(bndbox, "xmin").text = str(int(xmin))
                    ET.SubElement(bndbox, "ymin").text = str(int(ymin))
                    ET.SubElement(bndbox, "xmax").text = str(int(xmax))
                    ET.SubElement(bndbox, "ymax").text = str(int(ymax))
                    
        except Exception as e:
            print(f"处理文件 {yolo_path} 时出错: {e}")
            skipped_files += 1
            continue
        
        # 生成XML文件路径
        xml_path = os.path.join(output_dir, base_name + ".xml")
        
        # 生成格式化的XML字符串
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 移除XML声明（可选）
        pretty_xml = '\n'.join(pretty_xml.split('\n')[1:])
        
        # 保存XML文件
        with open(xml_path, 'w') as f:
            f.write(pretty_xml)
        
        converted_count += 1
        print(f"已转换: {yolo_path} -> {xml_path}")
    
    print("\n转换完成!")
    print(f"成功转换: {converted_count} 个标注文件")
    if missing_images > 0:
        print(f"警告: 有 {missing_images} 个标注文件找不到对应的图片")
    if skipped_files > 0:
        print(f"警告: 有 {skipped_files} 个文件因错误被跳过")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    yolo_labels_dir = "ssd-pytorch-master/VOCdevkit/VOC2007/yolotxt"  # YOLO标注文件目录
    images_dir = "ssd-pytorch-master/VOCdevkit/VOC2007/JPEGImages"  # 图片目录
    output_dir = "ssd-pytorch-master/VOCdevkit/VOC2007/Annotations"  # 输出XML目录
    
    # 执行转换
    convert_yolo_to_voc(yolo_labels_dir, images_dir, output_dir)