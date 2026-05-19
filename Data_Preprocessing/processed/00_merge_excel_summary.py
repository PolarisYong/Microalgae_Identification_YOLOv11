import os
import re  # 新增：正则匹配CH文件夹
import pandas as pd
from openpyxl import load_workbook


def merge_excel_sheets(ch_root_dir, output_file):
    # 原有核心逻辑完全不变：处理单个CH文件夹下的50个子文件夹
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 遍历50个子文件夹
        for i in range(1, 51):
            # 构建子文件夹名称，确保编号格式正确
            folder_name = f"IMG001x{i:03d}"
            folder_path = os.path.join(ch_root_dir, folder_name)

            # 检查子文件夹是否存在
            if not os.path.exists(folder_path):
                print(f"警告: 文件夹 {folder_path} 不存在，已跳过")
                continue

            # 构建results文件夹路径
            results_path = os.path.join(folder_path, "results")
            if not os.path.exists(results_path):
                print(f"警告: results文件夹 {results_path} 不存在，已跳过")
                continue

            # 获取results文件夹中的xlsx文件
            xlsx_files = [f for f in os.listdir(results_path) if f.endswith('.xlsx')]

            if not xlsx_files:
                print(f"警告: {results_path} 中未找到Excel文件，已跳过")
                continue

            # 假设每个results文件夹中只有一个xlsx文件
            xlsx_file = xlsx_files[0]
            xlsx_path = os.path.join(results_path, xlsx_file)
            sheet_name = "汇总统计"

            try:
                # 读取"汇总统计"页签内容
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

                # 检查是否包含必要的三列
                required_columns = ["图片名称", "目标数量", "总面积(μm²)"]
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    print(f"警告: {xlsx_file} 缺少必要列 {missing_columns}，已跳过")
                    continue

                # 检查"目标数量"列的最大值是否超过3
                max_target = df["目标数量"].max()
                if max_target <= 3:
                    print(f"跳过 {xlsx_file}: 目标数量最大值为 {max_target}，未超过3")
                    continue

                # 检查"目标数量"列是否存在0值
                if (df["目标数量"] == 0).any():
                    print(f"跳过 {xlsx_file}: 目标数量列中存在0值")
                    continue

                # 获取文件名（不含扩展名）作为新的页签名称
                new_sheet_name = os.path.splitext(xlsx_file)[0]

                # 将数据写入新的Excel文件，页签名为文件名
                df.to_excel(writer, sheet_name=new_sheet_name, index=False)
                print(f"已处理: {folder_name} -> {new_sheet_name} (目标数量最大值: {max_target})")

            except Exception as e:
                print(f"处理 {xlsx_path} 时出错: {str(e)}")

    print(f"✅ {ch_root_dir} 处理完成，结果已保存至 {output_file}\n")


if __name__ == "__main__":
    # 1. 修改为【父级根目录】（包含所有CH1/CH2...CH12的文件夹）
    parent_root = r"F:\Microalgae_Photoes\20260504"

    # 2. 输出文件的基础目录（自动生成CH1.xlsx/CH2.xlsx...）
    output_base_dir = r"F:\Microalgae_Photoes\20260504\数据汇总\01_原始数据"

    # 自动创建输出目录（防止目录不存在报错）
    os.makedirs(output_base_dir, exist_ok=True)

    # 3. 正则规则：匹配 CH + 1~2位数字 的文件夹（CH1/CH2...CH9/CH10/CH12）
    ch_pattern = re.compile(r'^CH\d{1,2}$')

    # 遍历父目录下所有文件夹，筛选符合条件的CH文件夹
    for folder_name in os.listdir(parent_root):
        folder_path = os.path.join(parent_root, folder_name)
        # 判断：是文件夹 + 名称匹配CH1~CH99格式
        if os.path.isdir(folder_path) and ch_pattern.match(folder_name):
            # 生成对应输出Excel路径
            output_excel = os.path.join(output_base_dir, f"{folder_name}.xlsx")
            # 执行合并（自动处理该CH文件夹下的所有数据）
            merge_excel_sheets(folder_path, output_excel)

    print("🎉 所有CH文件夹数据汇总全部完成！")