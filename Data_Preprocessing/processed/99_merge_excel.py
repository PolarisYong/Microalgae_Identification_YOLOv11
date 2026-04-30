import os
import pandas as pd


def merge_excel_files(folder_path, output_file):
    """
    合并文件夹中所有Excel文件的所有工作表到一个新的Excel文件
    """
    # 创建一个ExcelWriter对象用于写入汇总文件，指定引擎
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 只处理.xlsx文件
            if filename.endswith('.xlsx') and not filename.startswith('~$'):  # 排除临时文件和备份文件
                file_path = os.path.join(folder_path, filename)
                print(f"正在处理文件: {filename}")

                try:
                    # 读取当前Excel文件的所有工作表，明确指定引擎
                    excel_file = pd.ExcelFile(file_path, engine='openpyxl')
                    sheet_names = excel_file.sheet_names

                    # 遍历每个工作表并写入汇总文件
                    for sheet_name in sheet_names:
                        # 读取工作表数据
                        df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')

                        # 生成新的工作表名称，避免重复
                        new_sheet_name = f"{os.path.splitext(filename)[0]}_{sheet_name}"
                        # 工作表名称最长31个字符，超过则截断
                        if len(new_sheet_name) > 31:
                            new_sheet_name = new_sheet_name[:31]

                        # 将数据写入汇总文件的新工作表
                        df.to_excel(writer, sheet_name=new_sheet_name, index=False)
                        print(f"  已合并工作表: {sheet_name} -> {new_sheet_name}")
                except Exception as e:
                    print(f"  处理文件时出错: {str(e)}")

    print(f"\n所有文件合并完成，已保存至: {output_file}")


if __name__ == "__main__":
    # 文件夹路径（当前目录下的document文件夹）
    folder_path = r"F:\Microalgae_Photoes\summary\L100_20"
    # 输出文件路径
    output_file = r"F:\Microalgae_Photoes\summary\L100_20\summary.xlsx"

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
    else:
        # 检查是否有.xlsx文件
        xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') and not f.startswith('~$')]
        if not xlsx_files:
            print(f"警告: 文件夹 '{folder_path}' 中没有找到.xlsx文件")
        else:
            merge_excel_files(folder_path, output_file)