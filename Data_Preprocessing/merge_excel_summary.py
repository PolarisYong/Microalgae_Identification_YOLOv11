import os
import pandas as pd
from openpyxl import load_workbook


def merge_excel_sheets(root_dir, output_file):
    # 创建一个ExcelWriter对象，用于写入新的Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 遍历50个子文件夹
        for i in range(1, 51):
            # 构建子文件夹名称，确保编号格式正确
            folder_name = f"IMG001x{i:03d}"
            folder_path = os.path.join(root_dir, folder_name)

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

                # 检查"目标数量"列的最大值是否超过20
                max_target = df["目标数量"].max()
                if max_target <= 20:
                    print(f"跳过 {xlsx_file}: 目标数量最大值为 {max_target}，未超过20")
                    continue

                # 获取文件名（不含扩展名）作为新的页签名称
                new_sheet_name = os.path.splitext(xlsx_file)[0]

                # 将数据写入新的Excel文件，页签名为文件名
                df.to_excel(writer, sheet_name=new_sheet_name, index=False)
                print(f"已处理: {folder_name} -> {new_sheet_name} (目标数量最大值: {max_target})")

            except Exception as e:
                print(f"处理 {xlsx_path} 时出错: {str(e)}")

    print(f"所有处理已完成，结果已保存至 {output_file}")


if __name__ == "__main__":
    # 请在此处修改为您的根文件夹路径
    root_directory = r"F:\Microalgae_Photoes\20251002\Processed\001_L100_20"

    # 输出文件路径和名称
    output_excel = r"F:\Microalgae_Photoes\20251002\Processed\001_L100_20\CH1.xlsx"

    # 执行合并操作
    merge_excel_sheets(root_directory, output_excel)
