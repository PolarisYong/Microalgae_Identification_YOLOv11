#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd


def extract_umax_from_folder(root_path, output_path):
    """
    遍历指定文件夹下以日期命名的子文件夹，提取指定Excel中的多项数据。
    """
    # 用于存储最终结果的列表
    results = []

    # 正则表达式：严格匹配8位纯数字的日期文件夹 (例如: 20260504)
    date_pattern = re.compile(r'^\d{8}$')

    # 1. 遍历根目录下的所有子文件夹
    try:
        subfolders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    except Exception as e:
        print(f"❌ 无法读取根目录 '{root_path}': {e}")
        return

    for folder_name in subfolders:
        # 2. 校验文件夹名称是否为8位数字的日期格式
        if not date_pattern.match(folder_name):
            print(f"⏭️ 跳过非日期格式文件夹: {folder_name}")
            continue

        # 3. 拼接目标Excel文件的完整路径
        target_dir = os.path.join(root_path, folder_name, "数据汇总", "03_可视化结果", "96小时")

        if not os.path.exists(target_dir):
            print(f"⚠️ 路径不存在: {target_dir}")
            continue

        # 4. 遍历该目录下的所有 .xlsx 文件
        xlsx_files = [f for f in os.listdir(target_dir) if f.endswith('.xlsx')]

        if not xlsx_files:
            print(f"⚠️ 文件夹 '{folder_name}' 下的96小时目录中没有找到xlsx文件")
            continue

        for file_name in xlsx_files:
            file_path = os.path.join(target_dir, file_name)
            try:
                # 5. 读取Excel的指定页签
                df = pd.read_excel(file_path, sheet_name="汇总结果", header=0)

                # --- 【新增逻辑 1】查找“参与拟合的页签数量” ---
                fit_rows = df[df.iloc[:, 0].astype(str).str.contains("参与拟合的页签数量", na=False)]
                fit_count = fit_rows.iloc[0, 1] if not fit_rows.empty else None

                # --- 【原有逻辑】查找“最大比生长速率μmax (h^-1)” ---
                umax_rows = df[df.iloc[:, 0].astype(str).str.contains("最大比生长速率μmax", na=False)]
                umax_value = umax_rows.iloc[0, 1] if not umax_rows.empty else None

                # 如果两个指标都没找到，打印警告
                if fit_rows.empty and umax_rows.empty:
                    print(f"⚠️ 未找到任何目标数据行: {folder_name}/{file_name}")
                    continue

                # 6. 将结果追加到列表 (A, B, C, D 四列)
                results.append({
                    "日期": folder_name,
                    "文件名": file_name,
                    "参与拟合的页签数量": fit_count,  # C列
                    "最大比生长速率μmax (h^-1)": umax_value  # D列
                })

                print(f"✅ 成功提取: {folder_name} | {file_name} | 拟合数: {fit_count} | μmax: {umax_value}")

            except ValueError as ve:
                print(f"❌ 读取Excel页签失败 '{file_name}': 请检查是否存在'汇总结果'页签。详情: {ve}")
            except Exception as e:
                print(f"❌ 处理文件 '{file_name}' 时发生未知错误: {e}")

    # 7. 将所有结果汇总并导出为 all_data.xlsx
    if results:
        result_df = pd.DataFrame(results)
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_df.to_excel(output_path, index=False)
            print(f"\n🎉 汇总完成！共提取 {len(results)} 条数据。")
            print(f"📁 结果已保存至: {output_path}")
        except Exception as e:
            print(f"❌ 保存结果文件失败: {e}")
    else:
        print("\n⚠️ 没有提取到任何有效数据，未生成结果文件。")


# ================= 运行配置区 =================
if __name__ == "__main__":
    # 👉 请在这里修改你的【源数据根目录】路径
    ROOT_FOLDER = r"F:\Microalgae_Photoes\Archive"

    # 👉 请在这里修改你的【输出文件】完整路径
    OUTPUT_FILE = r"F:\Microalgae_Photoes\Archive\all_data.xlsx"

    # 执行提取函数
    extract_umax_from_folder(ROOT_FOLDER, OUTPUT_FILE)