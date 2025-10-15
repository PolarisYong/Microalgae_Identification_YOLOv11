import os
import re


def rename_json_files(root_dir):
    # 正则表达式模式：匹配完整的.json文件名结构
    # 分组说明：
    # 1: 通道编号 (数字部分)
    # 2: 小时数字 (XXX)
    # 3: 腔室数字 (XXX，来自IMG后的x后面部分)
    pattern = r'^channel(\d+)_(\d{3})_IMG\d+x(\d{3})\.json$'

    # 遍历所有文件夹和子文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 只处理json文件
            if filename.lower().endswith('.json'):
                # 尝试匹配模式
                match = re.match(pattern, filename)
                if match:
                    # 提取各部分信息并去除前导零
                    channel_num = match.group(1).lstrip('0') or '0'  # 通道号（如01→1）
                    hour_num = match.group(2).lstrip('0') or '0'  # 小时数（如008→8）
                    chamber_num = match.group(3).lstrip('0') or '0'  # 腔室号（如002→2）

                    # 构建新文件名
                    new_filename = f"CH{channel_num}_CB{chamber_num}_H{hour_num}.json"

                    # 构建完整路径
                    old_path = os.path.join(dirpath, filename)
                    new_path = os.path.join(dirpath, new_filename)

                    # 执行重命名
                    os.rename(old_path, new_path)
                    print(f"重命名: {filename} -> {new_filename}")
                else:
                    # 不匹配模式的文件将被跳过并提示
                    print(f"跳过不符合格式的文件: {filename}")


if __name__ == "__main__":
    # 使用当前工作目录作为根目录
    current_directory = r"F:\Microalgae Photoes\20251002\dataset_labelme"
    print(f"开始处理目录: {current_directory}")

    rename_json_files(current_directory)
    print("处理完成")
