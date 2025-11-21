import numpy as np
import pandas as pd

# 原始数据
data = {
    'Baseline': [0.6622, 0.6968, 0.6759, 0.7440, 0.4698],
    'SegNext_Attention': [0.6826, 0.7275, 0.6957, 0.7378, 0.4635],
    'TripletAttention': [0.6020, 0.7483, 0.6663, 0.7408, 0.4737],
    'CPCA': [0.6953, 0.6880, 0.6811, 0.7439, 0.4682],
    'CoTAttention': [0.6755, 0.7271, 0.6953, 0.7462, 0.4548],
    'GAM': [0.6581, 0.7445, 0.6972, 0.7486, 0.4596],
    'LocalWindowAttention': [0.6716, 0.7516, 0.7074, 0.7582, 0.4759]
}

# 转换为DataFrame
df = pd.DataFrame(data).T
df.columns = [f'指标{i + 1}' for i in range(df.shape[1])]


def entropy_weight_topsis(df):
    """使用熵权TOPSIS方法评估并选择最优方案"""
    # 数据标准化
    normalized_df = (df - df.min()) / (df.max() - df.min())

    # 计算熵值
    p = normalized_df / normalized_df.sum()
    p_log_p = p * np.log(p.replace(0, np.nan))
    entropy = -1 / np.log(len(normalized_df)) * p_log_p.sum()

    # # 检查是否有 NaN（如果某列全相同，熵值会为 0）
    # print("熵值计算结果（含指标5）:\n", entropy)

    # 计算权重
    weights = (1 - entropy) / (1 - entropy).sum()

    # 加权标准化矩阵
    weighted_normalized_df = normalized_df * weights

    # 计算理想解和负理想解
    ideal_best = weighted_normalized_df.max()
    ideal_worst = weighted_normalized_df.min()

    # 计算到理想解和负理想解的距离
    s_best = np.sqrt(((weighted_normalized_df - ideal_best) ** 2).sum(axis=1))
    s_worst = np.sqrt(((weighted_normalized_df - ideal_worst) ** 2).sum(axis=1))

    # 计算相对贴近度
    performance_score = s_worst / (s_best + s_worst)

    return {
        '原始数据': df,
        '标准化数据': normalized_df,
        '熵值': entropy,
        '权重': weights,
        '加权标准化数据': weighted_normalized_df,
        '到正理想解距离': s_best,
        '到负理想解距离': s_worst,
        '相对贴近度': performance_score
    }


# 执行评估
result = entropy_weight_topsis(df)

# 输出结果（分开显示）
print("1. 原始数据:")
print(result['原始数据'], "\n")

print("2. 标准化数据:")
print(result['标准化数据'].round(4), "\n")

print("3. 熵值:")
print(result['熵值'].round(4), "\n")

print("4. 指标权重:")
print(result['权重'].round(4), "\n")

print("5. 加权标准化数据:")
print(result['加权标准化数据'].round(4), "\n")

print("6. 到正理想解距离:")
print('\n')  # 空行
print(result['到正理想解距离'].round(4))

print("7. 到负理想解距离:")
print('\n')  # 空行
print(result['到负理想解距离'].round(4), "\n")

print("8. 相对贴近度（综合得分）:")
print(result['相对贴近度'].round(4), "\n")

# 找出最优方案
best_method = result['相对贴近度'].idxmax()
best_score = result['相对贴近度'][best_method]
print(f"最优注意力机制: {best_method}")
print(f"相对贴近度:{best_score:.4f}")

# 第一组数据
# data1 = {
#     'YOLOv11+Improved Neck（P3）': [0.6752, 0.7255, 0.6944, 0.7488, 0.4650],
#     'YOLOv11+Improved Neck（P4）': [0.7042, 0.6873, 0.6936, 0.7538, 0.4855],
#     'YOLOv11+Improved Neck（P5）': [0.6781, 0.7180, 0.6929, 0.7316, 0.4542],
#     'Ours': [0.6716, 0.7516, 0.7074, 0.7582, 0.4759]
# }
# df1 = pd.DataFrame(data1).T
# df1.columns = ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP50-95']
#
# result1 = entropy_weight_topsis(df1)

data2 = {
    'RT-DETR': [0.6565, 0.7054, 0.6801, 0.6828, 0.4375],
    'TOOD': [0.6330, 0.7090, 0.6688, 0.6960, 0.4460],
    'Faster R-CNN': [0.6280, 0.7070, 0.6652, 0.6980, 0.4270],
    'CascadeR-CNN': [0.5810, 0.7300, 0.6470, 0.6800, 0.4340],
    'RetinaNet': [0.5370, 0.7240, 0.6166, 0.6270, 0.3750],
    'RTMDet': [0.6280, 0.7040, 0.6638, 0.7050, 0.4440],
    'YOLOv3-tiny': [0.5486, 0.6669, 0.6020, 0.6432, 0.3265],
    'YOLOv5': [0.6531, 0.6906, 0.6713, 0.7236, 0.4646],
    'YOLOv6': [0.7333, 0.6307, 0.6781, 0.7115, 0.4448],
    'YOLOv8': [0.6573, 0.7267, 0.6903, 0.7259, 0.4496],
    'YOLOv9': [0.6610, 0.7208, 0.6896, 0.7252, 0.4581],
    'YOLOv10': [0.6772, 0.6505, 0.6636, 0.7078, 0.4373],
    'YOLOv12': [0.6235, 0.7626, 0.6861, 0.7354, 0.4621],
    'Ours': [0.6716, 0.7516, 0.7094, 0.7582, 0.4759]
}
df2 = pd.DataFrame(data2).T
df2.columns = ['Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP50-90']

result2 = entropy_weight_topsis(df2)

