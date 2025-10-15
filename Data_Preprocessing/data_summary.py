import os
import pandas as pd


def process_excel_file(file_path, channel):
    """处理单个Excel文件，提取所需数据"""
    try:
        # 读取"汇总统计"页签
        df = pd.read_excel(file_path, sheet_name="汇总统计")

        # 检查必要的列是否存在
        required_columns = ["图片名称", "目标数量", "总面积(μm²)"]
        if not all(col in df.columns for col in required_columns):
            print(f"警告: {file_path} 中缺少必要的列，已跳过")
            return None

        # 提取数据并去除空值
        target_counts = df["目标数量"].dropna().tolist()
        total_areas = df["总面积(μm²)"].dropna().tolist()

        if not target_counts or not total_areas:
            print(f"警告: {file_path} 中没有有效数据，已跳过")
            return None

        # 计算所需指标
        initial_count = target_counts[0]
        initial_area = total_areas[0]
        final_count = target_counts[-1]
        final_area = total_areas[-1]

        max_count = max(target_counts)
        max_count_time = target_counts.index(max_count) + 1 # 行数减一（索引从0开始）

        max_area = max(total_areas)
        max_area_time = total_areas.index(max_area) + 1 # 行数减一（索引从0开始）

        # 提取腔室信息 (CB1, CB2, ..., CB50)
        chamber = os.path.splitext(os.path.basename(file_path))[0].split("_")[1]

        return {
            "腔室": f"{channel}_{chamber}",  # 例如 CH1_CB1
            "初始细胞数量": initial_count,
            "初始细胞面积(μm²)": initial_area,
            "最终细胞数量": final_count,
            "最终细胞面积(μm²)": final_area,
            "最多细胞数量": max_count,
            "最多细胞出现的时刻": f"H{max_count_time}",
            "最大细胞面积(μm²)": max_area,
            "最大细胞面积出现的时刻": f"H{max_area_time}"
        }

    except Exception as e:
        print(f"处理 {file_path} 时出错: {str(e)}")
        return None


def main():
    # 定义文件夹路径
    processed_dir = r"F:\Microalgae_Photoes\20251002\Processed"

    # 检查processed文件夹是否存在
    if not os.path.exists(processed_dir):
        print(f"错误: 文件夹 '{processed_dir}' 不存在")
        return

    # 获取所有流道文件夹 (001_L100_20, 002_..., 等)
    flow_dirs = [d for d in os.listdir(processed_dir)
                 if os.path.isdir(os.path.join(processed_dir, d))
                 and d.split("_")[0].isdigit()]

    if not flow_dirs:
        print(f"错误: 在 '{processed_dir}' 中未找到任何流道文件夹")
        return

    # 创建一个字典存储所有流道的DataFrame
    all_data = {}

    # 处理每个流道文件夹
    for flow_dir in flow_dirs:
        flow_path = os.path.join(processed_dir, flow_dir)
        flow_id = flow_dir.split("_")[0]  # 提取001, 002等流道编号
        channel = f"CH{flow_id.lstrip('0')}"  # 转换为CH1, CH2等

        print(f"正在处理流道 {flow_id}...")

        # 收集该流道的所有数据
        flow_data = []

        # 处理每个腔室文件夹 (IMG001x001 到 IMG001x050)
        for chamber_num in range(1, 51):  # 1到50个腔室
            # 构建腔室文件夹名称 (例如 IMG001x001)
            img_prefix = f"IMG001x"
            chamber_folder = f"{img_prefix}{chamber_num:03d}"
            chamber_path = os.path.join(flow_path, chamber_folder)

            # 检查腔室文件夹是否存在
            if not os.path.exists(chamber_path):
                print(f"警告: 腔室文件夹 {chamber_path} 不存在，已跳过")
                continue

            # 查找results文件夹
            results_path = os.path.join(chamber_path, "results")
            if not os.path.exists(results_path):
                print(f"警告: results文件夹 {results_path} 不存在，已跳过")
                continue

            # 查找Excel文件 (例如 CH1_CB1.xlsx)
            excel_filename = f"{channel}_CB{chamber_num}.xlsx"
            excel_path = os.path.join(results_path, excel_filename)

            if not os.path.exists(excel_path):
                print(f"警告: Excel文件 {excel_path} 不存在，已跳过")
                continue

            # 处理Excel文件
            chamber_data = process_excel_file(excel_path, channel)
            if chamber_data:
                flow_data.append(chamber_data)

        # 将该流道的数据存储为DataFrame
        if flow_data:
            all_data[flow_id] = pd.DataFrame(flow_data)
            print(f"已收集流道 {flow_id} 的数据，共 {len(flow_data)} 条记录")
        else:
            print(f"警告: 流道 {flow_id} 没有有效的数据")

    # 所有数据处理完成后，统一写入Excel文件
    output_file = r"F:\Microalgae_Photoes\20251002\Processed\数据总览.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for flow_id, df in all_data.items():
            df.to_excel(writer, sheet_name=flow_id, index=False)
            print(f"已将流道 {flow_id} 的数据写入到页签 {flow_id}")

    print(f"数据汇总完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
