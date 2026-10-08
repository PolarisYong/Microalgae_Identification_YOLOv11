import pandas as pd
import re
import os

# ================== 用户直接修改这里 ==================
input_path = r"F:\Microalgae_Photoes\Archive\all_data.xlsx"  # 输入文件路径（原文件）
output_path = r"F:\Microalgae_Photoes\Archive\all_data_3data.xlsx"  # 输出文件路径（新文件）


# ==================================================

def process_excel(input_path, output_path):
    """
    处理Excel文件：根据D列生成三列新数据（F/G/H列）

    参数:
    input_path (str): 输入Excel文件路径
    output_path (str): 输出Excel文件路径
    """
    # 读取Excel文件
    df = pd.read_excel(input_path)

    # 确保D列存在（按列位置处理：第4列=A(0),B(1),C(2),D(3)）
    if len(df.columns) < 4:
        raise ValueError("Excel文件必须包含至少4列（A/B/C/D）")

    # 获取D列数据（按位置索引，避免列名不匹配问题）
    d_col_data = df.iloc[:, 3]  # 第4列（0-based索引为3）

    # 解析D列并生成新列数据
    light_intensity = []
    ammonia_nitrogen = []
    inorganic_carbon = []

    for value in d_col_data:
        # 处理空值
        if pd.isna(value):
            light_intensity.append(None)
            ammonia_nitrogen.append(None)
            inorganic_carbon.append(None)
            continue

        # 清理并标准化字符串（处理特殊连字符U+2011和普通连字符）
        clean_value = str(value).replace('\u2011', '-').strip()

        # 正则匹配：L120-N160-IC5.25% 或 L30-N20-IC0.5%-OC2 等格式
        pattern = r'L(\d+)-N(\d+)-IC([\d.]+)%?'
        match = re.search(pattern, clean_value)

        if not match:
            # 匹配失败时填入None
            light_intensity.append(None)
            ammonia_nitrogen.append(None)
            inorganic_carbon.append(None)
            continue

        # 提取关键数值
        l_val = int(match.group(1))
        n_val = int(match.group(2))
        ic_val = float(match.group(3))

        # 计算Light intensity (F列)
        if l_val == 210:
            light_intensity.append(1)
        elif l_val == 120:
            light_intensity.append(0)
        elif l_val == 30:
            light_intensity.append(-1)
        else:
            light_intensity.append(None)  # 非定义值

        # 计算Ammonia nitrogen (G列)
        if n_val == 300:
            ammonia_nitrogen.append(1)
        elif n_val == 160:
            ammonia_nitrogen.append(0)
        elif n_val == 20:
            ammonia_nitrogen.append(-1)
        else:
            ammonia_nitrogen.append(None)  # 非定义值

        # 计算Inorganic carbon (H列)
        if ic_val == 10:
            inorganic_carbon.append(1)
        elif ic_val == 5.25:
            inorganic_carbon.append(0)
        elif ic_val == 0.5:
            inorganic_carbon.append(-1)
        else:
            inorganic_carbon.append(None)  # 非定义值

    # 在D列后插入新列（确保新列位于Excel的F/G/H列位置）
    insert_pos = 5  # F列是第6列(0-based索引=5)
    df.insert(insert_pos - 1, "Light intensity", light_intensity)  # F列 (索引5)
    df.insert(insert_pos, "Ammonia nitrogen", ammonia_nitrogen)  # G列 (索引6)
    df.insert(insert_pos + 1, "Inorganic carbon", inorganic_carbon)  # H列 (索引7)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 保存到新文件
    df.to_excel(output_path, index=False)
    return output_path


if __name__ == "__main__":
    # 直接使用代码中定义的路径
    output_file = process_excel(input_path, output_path)

    print(f"✅ 处理完成！新文件已保存至: {output_file}")
    print("\n📌 新增列说明:")
    print("   F列 [Light intensity]: L值→210=1, 120=0, 30=-1")
    print("   G列 [Ammonia nitrogen]: N值→300=1, 160=0, 20=-1")
    print("   H列 [Inorganic carbon]: IC值→10%=1, 5.25%=0, 0.5%=-1")
    print("\n💡 提示：要修改路径，请直接编辑脚本开头的 input_path 和 output_path 变量")