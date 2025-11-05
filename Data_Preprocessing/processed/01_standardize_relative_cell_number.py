import pandas as pd
import openpyxl


def standardize_cell_data(input_file, output_file):
    # 读取Excel文件，获取所有页签名称
    excel_file = pd.ExcelFile(input_file)
    sheet_names = excel_file.sheet_names

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
                df['目标数量'] = (df['目标数量'] / initial_count).round(4)

            # 总面积列标准化
            if initial_area == 0:
                df['总面积(μm²)'] = 0.0
                print(f"ℹ️  提示：页签【{sheet}】'总面积(μm²)'初始值为0，标准化后所有值设为0")
            else:
                df['总面积(μm²)'] = (df['总面积(μm²)'] / initial_area).round(4)

            # 新增：计算相对平均细胞面积
            if initial_avg_area == 0:
                # 处理初始平均面积为0的特殊情况
                df['相对平均细胞面积'] = 0.0
            else:
                # 计算当前平均面积与初始平均面积的比值
                # 使用where避免除以0错误
                current_avg_area = df_tmp['总面积(μm²)'] / df_tmp['目标数量'].where(df['目标数量'] != 0, 1)
                df['相对平均细胞面积'] = (current_avg_area / initial_avg_area).round(4)

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


# 配置文件路径
INPUT_EXCEL = r"F:\Microalgae_Photoes\20251104\数据处理结果\原始数据\CH6.xlsx"  # 原始数据文件路径
OUTPUT_EXCEL = r"F:\Microalgae_Photoes\20251104\数据处理结果\标准化数据\CH6_标准化.xlsx"
# OUTPUT_EXCEL = INPUT_EXCEL[:-5] + "_标准化.xlsx"

# 执行标准化
if __name__ == "__main__":
    standardize_cell_data(INPUT_EXCEL, OUTPUT_EXCEL)