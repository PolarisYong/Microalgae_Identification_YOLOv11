import pandas as pd
import openpyxl
import re  # 新增正则匹配
import os


def standardize_cell_data(input_file, output_file, skip_sheet_name):
    # 读取Excel文件，获取所有页签名称
    excel_file = pd.ExcelFile(input_file)
    sheet_names = excel_file.sheet_names
    # 过滤掉需要跳过的页签
    sheet_names = [name for name in sheet_names if name not in skip_sheet_name]
    # 存储所有处理后的sheet数据，用于后续汇总
    processed_sheets = []

    # 创建Excel写入器，用于保存标准化后的数据
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet in sheet_names:
            # 读取当前页签的原始数据
            df = pd.read_excel(input_file, sheet_name=sheet)
            df_tmp = df.copy(deep=True)
            # 检查必要列是否存在，若缺少则跳过该页签
            required_columns = ['图片名称', '目标数量', '总面积(μm²)']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️  警告：页签【{sheet}】缺少必要列，已跳过该页签")
                continue

            # 获取初始值（第一行数据）
            initial_count = df.iloc[0]['目标数量']
            initial_area = df.iloc[0]['总面积(μm²)']

            # 计算初始平均细胞面积
            if initial_count == 0:
                initial_avg_area = 0
                print(f"ℹ️  提示：页签【{sheet}】初始目标数量为0，无法计算初始平均细胞面积")
            else:
                initial_avg_area = initial_area / initial_count

            # 目标数量列标准化
            if initial_count == 0:
                df['目标数量'] = 0.0
                print(f"ℹ️  提示：页签【{sheet}】'目标数量'初始值为0，标准化后所有值设为0")
            else:
                df['目标数量'] = (df['目标数量'] / initial_count).round(2)

            # 总面积列标准化
            if initial_area == 0:
                df['总面积(μm²)'] = 0.0
                print(f"ℹ️  提示：页签【{sheet}】'总面积(μm²)'初始值为0，标准化后所有值设为0")
            else:
                df['总面积(μm²)'] = (df['总面积(μm²)'] / initial_area).round(2)

            # 新增：计算相对平均细胞面积
            if initial_avg_area == 0:
                # 处理初始平均面积为0的特殊情况
                df['相对平均细胞面积'] = 0.0
            else:
                # 计算当前平均面积与初始平均面积的比值
                # 使用where避免除以0错误
                current_avg_area = df_tmp['总面积(μm²)'] / df_tmp['目标数量'].where(df['目标数量'] != 0, 1)
                df['相对平均细胞面积'] = (current_avg_area / initial_avg_area).round(2)

            # 将标准化后的数据写入新Excel的对应页签
            df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"✅ 已完成页签【{sheet}】的标准化处理")

            # 保存处理后的sheet数据用于汇总
            processed_sheets.append(df)

        # 生成数据汇总sheet
        if processed_sheets:
            # 初始汇总数据为空
            summary_df = pd.DataFrame()
            for i, sheet_df in enumerate(processed_sheets):
                # 每个sheet数据前添加两列空列（除了第一个sheet）
                if i > 0:
                    empty_cols = pd.DataFrame(columns=[f'空列{i}_1', f'空列{i}_2'])
                    summary_df = pd.concat([summary_df, empty_cols, sheet_df], axis=1)
                else:
                    # 第一个sheet前面不需要空列
                    summary_df = pd.concat([summary_df, sheet_df], axis=1)

            # 将汇总数据写入新sheet
            summary_df.to_excel(writer, sheet_name='数据汇总', index=False)
            print(f"✅ 已生成【数据汇总】页签")

    print(f"\n🎉 所有有效页签处理完成！标准化结果已保存至：{output_file}")


# ===================== 批量处理核心配置（仅修改这里）=====================
if __name__ == "__main__":
    # 1. 原始数据目录（存放所有CH1.xlsx/CH2.xlsx的文件夹）
    raw_data_dir = r"F:\Microalgae_Photoes\20260504\数据汇总\01_原始数据"
    # 2. 标准化输出目录（自动生成CH1_标准化.xlsx...）
    standard_output_dir = r"F:\Microalgae_Photoes\20260504\数据汇总\02_标准化数据"
    # 3. 需要跳过的页签（保持你原来的配置不变）
    skip_sheet = {"数据汇总", "← 👈25-9-26批  25-10-25批👉→", "👉→理论预测 9组"}

    # 自动创建输出目录
    os.makedirs(standard_output_dir, exist_ok=True)
    # 正则匹配：CH1.xlsx ~ CH99.xlsx
    excel_pattern = re.compile(r'^CH\d{1,2}\.xlsx$')

    # 遍历原始数据目录，批量处理所有CH文件
    for file_name in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file_name)
        # 筛选：是文件 + 名称匹配CHN.xlsx
        if os.path.isfile(file_path) and excel_pattern.match(file_name):
            # 动态生成输出文件名：CH4.xlsx → CH4_标准化.xlsx
            ch_name = os.path.splitext(file_name)[0]
            output_file = os.path.join(standard_output_dir, f"{ch_name}_标准化.xlsx")

            print(f"\n=====================================")
            print(f"开始处理：{file_name}")
            print(f"输出文件：{os.path.basename(output_file)}")
            print(f"=====================================\n")

            # 调用原有函数处理
            standardize_cell_data(file_path, output_file, skip_sheet)

    print("\n🎉🎉🎉 所有CH文件标准化处理全部完成！")